import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, rope_scaling={'type': 'linear', 'factor': 32.0}).cpu()
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

model_id = "/Users/antoniogrotta/Downloads/lora_model"
LLM = pipeline("text-generation", model=model_id)
prompt = f"### Instruction:\nSuggest the next physiotherapy exercise.\n\n###" \
         f"Input:\nExercise History:\nexercise: Clamshell, feedback: 0.2\nexercise: Heel Slides, feedback: 0.4\nexercise: Wall Slides, feedback: 0.6000000000000001\nexercise: Bridging, feedback: 0.1\nexercise: Seated Knee Extension, feedback: 0.1\nexercise: Hamstring Stretch, feedback: 0.8\n\nPatient Data:\nage: 68\ninjury_type: Ankle sprain\n\n### Response:"
result = LLM(prompt, max_length=210, truncation=True)
print(result)
