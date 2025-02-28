import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist
import torch

model_path = "/scratch/reference/ai/models/LLMs/deepseek-r1"
cache_dir = "/project/s10002/model"

print("Gloo available:", dist.is_gloo_available())

# Load tokenizer (tokenizers don't need to be on a device)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir, trust_remote_code=True)

# Load model with ZeRO-Inference
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    cache_dir=cache_dir,
    torch_dtype=torch.float32
)

# Initialize DeepSpeed with CPU offloading
ds_engine = deepspeed.init_inference(
    model=model,
    config="ds_config.json"
)

model = ds_engine.module  # Assign the DeepSpeed model back

# Run inference
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))


if dist.get_rank() == 0:
    with open("result.txt", "w") as f:
        f.write(outputs)

    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
