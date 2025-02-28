#!/bin/bash
#SBATCH --job-name=FSDP_LLM
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1  # One process per node
#SBATCH --cpus-per-task=192
#SBATCH --mem=350G
#SBATCH --time=02:00:00
#SBATCH -A s10002
#SBATCH --output=fsdp-%j.out
#SBATCH --error=fsdp-%j.err

# Load necessary modules
module load python
module load mysw
module load pytorch

# Run distributed inference with FSDP
torchrun --nnodes=8 --nproc_per_node=1 fsdp_inference.py
