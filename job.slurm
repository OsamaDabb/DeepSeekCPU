#!/bin/bash
#SBATCH --job-name=deepseek_inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=output.log

source venv/bin/activate
deepspeed --num_cpus=16 inference.py --deepspeed ds_config.json
