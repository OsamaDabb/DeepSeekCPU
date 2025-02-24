import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "deepseek-ai/deepseek-llm-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype="float32"
)

inputs = tokenizer("Hello, how can I help you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
