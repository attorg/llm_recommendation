from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import time

model_id = "meta-llama/Meta-Llama-3-8B"
# model_id = "meta-llama/Llama-2-7b-hf"

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

tokenizer=AutoTokenizer.from_pretrained(model_id)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

start_time = time.time()
generated_text = generator("Hey, how are you doing today?")
end_time = time.time()
execution_time = end_time - start_time

print(f"Generation time: {execution_time:.2f} seconds")
print(generated_text)
