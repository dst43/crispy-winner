#!/bin/bash
#SBATCH -J moe_expert
#SBATCH --time=1-00:00:00
#SBATCH -o %x_%j_%n.out
#SBATCH -e %x_%j_%n.err
#SBATCH -p cas_v100nv_4

#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --comment pytorch
#SBATCH --cpus-per-gpu=10

srun -N2 -o %x_%j_%n.out -e %x_%j_%n.err moe_expert_run.sh

exit 0
