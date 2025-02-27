#!/bin/bash
#SBATCH -N 1
#SBATCH -J Inference-Test
#SBATCH --mail-user=osama.dabbousi@kaust.edu.sa
#SBATCH --mail-type=NONE
#SBATCH -A s10002
#SBATCH -t 01:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1

module load python
module load mysw
module load pytorch

#run the application:
srun --hint=nomultithread \
     --ntasks=1 \
     --cpus-per-task=192 \
     --mem=2.2TB \
     --cpu-bind=threads \
     deepspeed --num_gpus 0 inference.py --deepspeed ds_config.json
