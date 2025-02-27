import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
from torch import device

model_path = "/scratch/reference/ai/models/LLMs/deepseek-r1"
cache_dir = "/project/s10002/model"
cpu_device = torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True).to(cpu_device)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    cache_dir=cache_dir,
    device_map="cpu",
    torch_dtype="float32"
).to(cpu_device)

inputs = tokenizer("Hello, how can I help you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)

if dist.get_rank() == 0:
    with open("result.txt", "w") as f:
        f.write(outputs)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
