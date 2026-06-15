# evaluate_lora.py
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

from prompts import TEST_PROMPTS

BASE_PATH = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "./train/lora_adapter"

tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_PATH,
    quantization_config=bnb_config,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, LORA_PATH)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("LoRA Fine-Tuned Model Evaluation:")
for prompt in TEST_PROMPTS:
    output = pipe(prompt, max_new_tokens=100)[0]['generated_text']
    print(f"\n🔹 Prompt: {prompt}\nOutput:\n{output}")
