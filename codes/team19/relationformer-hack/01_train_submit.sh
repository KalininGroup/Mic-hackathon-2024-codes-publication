#!/bin/bash
#SBATCH --time=05-00:00:00      # Job time allocation
#SBATCH --gres=gpu:4            # Request GPUs
### SBATCH --constraint=ampere|volta       # Request specific nodes
#SBATCH --mem=64G               # Memory
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH -c 16                 # Number of cores per task/node
#SBATCH -J 01_train    # Job name
#SBATCH -o train_%j.out          # Output file

# Load environment
# module load anaconda # On old triton
module load mamba gcc # On new triton
module load cuda
source activate /scratch/phys/project/sin/hackathon/.conda_envs/hack
export OMP_NUM_THREADS=1

# Set SLURM_NTASKS
# Set SLURM_NTASKS
export SLURM_NTASKS=${SLURM_NPROCS:-1}

# Print job info
echo "Job ID: "$SLURM_JOB_ID
echo "Job Name: "$SLURM_JOB_NAME
echo "Running on nodes: "$SLURM_JOB_NODELIST
echo "which Python: "$(which python)

python train.py --config configs/synth_3D.yaml --cuda_visible_device 0 1 2 3  --nproc_per_node 4

