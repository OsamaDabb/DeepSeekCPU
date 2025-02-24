import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist

model_path = "deepseek-ai/deepseek-llm-7b-chat"
cache_dir = "/project/s10002/model"

tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    cache_dir=cache_dir,
    device_map="cpu",
    torch_dtype="float32"
)

inputs = tokenizer("Hello, how can I help you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

if dist.get_rank() == 0:
    with open("result.txt", "w") as f:
        f.write(outputs)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
