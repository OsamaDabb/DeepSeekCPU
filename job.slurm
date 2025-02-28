#!/bin/bash
#SBATCH -N 2
#SBATCH -p ppn
#SBATCH --mem=2T
#SBATCH -J Inference-Test
#SBATCH --mail-user=osama.dabbousi@kaust.edu.sa
#SBATCH --mail-type=NONE
#SBATCH -A s10002
#SBATCH -t 01:00:00

# OpenMP settings:
export OMP_NUM_THREADS=1
PROCESS_COUNT=2

# Load necessary modules
module load python
module load mysw
module load pytorch

# Set MASTER_ADDR and MASTER_PORT for multi-node communication
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500  # Choose an unused port

# Generate a hostfile for DeepSpeed manually
scontrol show hostnames $SLURM_NODELIST | awk '{print $1, "slots=1"}' > hostfile

# Determine node rank (DeepSpeed requires unique ranks per node)
export NODE_RANK=$SLURM_NODEID  # Unique rank for each node

# Debug: Print assigned nodes and node rank
echo "Master Node: $MASTER_ADDR"
echo "Node List: $(cat hostfile)"
echo "Node Rank: $NODE_RANK"

# Run DeepSpeed separately on each node with correct parameters
srun --hint=nomultithread \
     --ntasks=$PROCESS_COUNT \
     --cpus-per-task=192 \
     --cpu-bind=threads \
     deepspeed --hostfile=hostfile --no_ssh \
     --num_nodes=$PROCESS_COUNT --node_rank=$NODE_RANK \
     --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
     inference.py --deepspeed ds_config.json
