# evaluate_base.py
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from prompts import TEST_PROMPTS

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("Base Model Evaluation:")
for prompt in TEST_PROMPTS:
    output = pipe(prompt, max_new_tokens=100)[0]['generated_text']
    print(f"\n🔹 Prompt: {prompt}\nOutput:\n{output}")
