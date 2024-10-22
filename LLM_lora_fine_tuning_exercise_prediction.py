import torch
import wandb
import os
import numpy as np
from sklearn.metrics import accuracy_score
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import pipeline


# Load the JSON dataset and convert it into a Dataset type
def load_json_dataset(json_path):
    dataset = load_dataset('json', data_files=json_path)
    return dataset


def create_prompt(example):
    instruction = example['instruction']
    exercise_history = example['input']['exercise_history']
    exercise_details = f"\n".join([f"exercise: {ex['name']}, feedback: {ex['feedback']}" for ex in exercise_history])

    patient_data = example['input']['patient_data']
    patient_details = f"\n".join([f"{key}: {value}" for key, value in patient_data.items()])

    input_text = f"Exercise History:\n{exercise_details}\n\nPatient Data:\n{patient_details}"

    # Combine 'instruction' and 'input' into a 'prompt'
    example[
        'prompt'] = f"Below is an instruction that describes a task, paired with an input that provides further context. " \
                    f"The input is composed by an exercise history, whcih is a list of exercises and feedback given by patients." \
                    f"Write a response that appropriately completes the request.\n\n " \
                    f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    # Keep 'output' as it is
    example['output'] = example['output']
    return example


# Tokenization function for instruction-based dataset
def tokenize_function(example):
    prompt_tokens = tokenizer(example['prompt'], padding='max_length', max_length=200, truncation=True)
    output_tokens = tokenizer(example['output'], padding='max_length', max_length=10, truncation=True)
    # Merge both tokenized results
    return {
        'input_ids': prompt_tokens['input_ids'],
        'attention_mask': prompt_tokens['attention_mask'],
        'labels': output_tokens['input_ids'],  # Typically, labels are used for training
    }


def compute_metrics(eval_pred):
    # Unpack the model predictions and true labels
    logits, labels = eval_pred

    # Get the predicted token IDs (argmax of logits)
    predictions = np.argmax(logits, axis=-1)

    # Flatten the labels and predictions to compute accuracy
    # This assumes that padding tokens are not -100 (common in language models)
    true_labels = labels.flatten()
    pred_labels = predictions.flatten()

    # Filter out padding tokens (-100 in the labels)
    mask = true_labels != -100
    filtered_true_labels = true_labels[mask]
    filtered_pred_labels = pred_labels[mask]

    # Compute accuracy
    accuracy = accuracy_score(filtered_true_labels, filtered_pred_labels)

    return {'accuracy': accuracy}


wandb.login(key='d02a82043c8e6a3936c1cdac10501c0e17272b2f')
wandb.init(project="exercise_recommendation_lora_fine_tuning", name="exercise_recommendation_run")

# Load tokenizer for LLaMA
# model_id = "meta-llama/Meta-Llama-3-8B"
# model_id = "meta-llama/Llama-2-7b-hf"
model_id = "openai-community/gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

dataset = load_json_dataset("/Users/antoniogrotta/repositories/llm_recommendation/data/exercise_examples.json")
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)

dataset = dataset.map(create_prompt)

'''
max_length = 0
for example in dataset['train']:
    tokenized_output = tokenizer(example['prompt'], truncation=False)
    example_length = len(tokenized_output['input_ids'])
    max_length = max(max_length, example_length)

print(f"The maximum length is: {max_length}")  # 197

max_length = 0
for example in dataset['test']:
    tokenized_output = tokenizer(example['prompt'], truncation=False)
    example_length = len(tokenized_output['input_ids'])
    max_length = max(max_length, example_length)

print(f"The maximum length is: {max_length}")  # 188

max_length = 0
for example in dataset['train']:
    tokenized_output = tokenizer(example['output'], truncation=False)
    example_length = len(tokenized_output['input_ids'])
    max_length = max(max_length, example_length)

print(f"The maximum length is: {max_length}")  # 5
'''

# Remove unnecessary columns
dataset = dataset.remove_columns(['instruction', 'input'])
dataset['train'] = Dataset.from_dict({
    'prompt': dataset['train']['prompt'],
    'output': dataset['train']['output']
})

dataset['test'] = Dataset.from_dict({
    'prompt': dataset['test']['prompt'],
    'output': dataset['test']['output']
})

# Apply the tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Prepare train and eval datasets
train_dataset = tokenized_dataset['train']
eval_dataset = tokenized_dataset['test']

# Prepare the model for LoRA training
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(model_id,
                                             quantization_config=config,
                                             device_map="auto")
model = prepare_model_for_kbit_training(model)

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # LoRA rank
    lora_alpha=16,  # LoRA scaling
    target_modules=['q_proj', 'v_proj'],  # Specific layers to apply LoRA
    lora_dropout=0.1,  # Dropout for LoRA
    bias='none'
)
# Apply LoRA to the model
model = get_peft_model(model, lora_config)

'''
# set the wandb project where this run will be logged
os.environ["WANDB_PROJECT"]="exercise_recommendation_lora_fine_tuning"

os.environ["WANDB_LOG_MODEL"] = "end" 

# turn off watch to log faster
os.environ["WANDB_WATCH"]="false"
'''

# Define training arguments
training_args = TrainingArguments(
    output_dir="llama_lora_trainer",
    report_to="wandb",
    eval_strategy="epoch",
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=1,  # Log every 10 steps
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("model/ft_llama_lora_model")
tokenizer.save_pretrained("model/ft_llama_lora_model")

# Load the model using pipeline for text generation
classifier = pipeline("text-generation", model="model/ft_llama_lora_model")
result = classifier("add prompt", max_length=50)
print(result)
