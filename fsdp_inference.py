import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from transformers import AutoModelForCausalLM, AutoTokenizer


def setup_distributed():
    """Initializes distributed process group."""
    dist.init_process_group(backend="gloo", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Rank {rank}/{world_size} process initialized.")
    return rank


def load_sharded_model(model_name="big_model"):
    """Loads the model with FSDP wrapping before full instantiation."""
    # Define an automatic wrapping policy (shards layers before full load)
    def auto_wrap_policy(module, recurse, nonwrap):
        return size_based_auto_wrap_policy(
            module, recurse, nonwrap, min_num_params=50_000_000)  # 50M param threshold

    # Load model in sharded manner
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,  # Prevents full model instantiation in RAM
        torch_dtype=torch.float32  # Adjust dtype if needed
    )

    # Wrap model in FSDP before weights are fully materialized
    model = FSDP(model, auto_wrap_policy=auto_wrap_policy)

    return model


if __name__ == "__main__":
    rank = setup_distributed()

    # Load tokenizer (no need to shard)
    tokenizer = AutoTokenizer.from_pretrained("big_model")

    # Load and shard model before full instantiation
    model = load_sharded_model().to("cpu")

    # Example inference
    inputs = tokenizer("Hello, world!", return_tensors="pt")
    outputs = model.generate(**inputs)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Clean up
    dist.destroy_process_group()
