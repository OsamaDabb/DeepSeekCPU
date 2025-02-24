#!/bin/bash
#SBATCH --time 01:0:0
#SBATCH --partition=ppn

source $MY_SW/miniconda3-amd64/bin/activate

conda activate $MY_SW/envs/deepseek

conda install conda-forge::deepspeed
conda install conda-forge::transformers
