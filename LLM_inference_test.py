import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import time

# model_id = "meta-llama/Meta-Llama-3-8B"
# model_id = "/Users/antoniogrotta/repositories/LLM_finetuning_exercise/meta-llama/Meta-Llama-3-8B"
# model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).cpu()

# model_id = "meta-llama/Llama-2-7b-hf"
model_id = "openai-community/gpt2"
# model_id = "/Users/antoniogrotta/repositories/LLM_finetuning_exercise/meta-llama/Llama-2-7b-hf"

generator = transformers.pipeline(
    "text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

start_time = time.time()
generated_text = generator("Hey, how are you doing today?")
end_time = time.time()
execution_time = end_time - start_time
print(f"Generation time: {execution_time:.2f} seconds")
print(generated_text)

