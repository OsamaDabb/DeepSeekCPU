#!/bin/bash
#SBATCH -N 1
#SBATCH -J Inference-Test
#SBATCH --mail-user=osama.dabbousi@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH -A s10002
#SBATCH -t 01:00:00

#OpenMP settings:
export OMP_NUM_THREADS=1

module load python
module load mysw

#run the application:
srun --hint=nomultithread \
     --ntasks=192 \
     --ntasks-per-node=192 \
     --ntasks-per-socket=96 \
     --cpus-per-task=1 \
     --ntasks-per-core=1 \
     --mem-bind=v,local \
     --cpu-bind=threads \
     deepspeed --num_nodes 1 --num_gpus 0 inference.py --deepspeed ds_config.json
