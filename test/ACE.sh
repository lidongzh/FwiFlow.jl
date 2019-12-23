#!/bin/bash
#SBATCH --job-name=ACE
#SBATCH --nodes=1 --ntasks=1
#SBATCH --output=ACE_%A_%a.out
#SBATCH --error=ACE_%A_%a.err
#SBATCH --partition CPU
#SBATCH --array=1-8
echo "$SLURM_ARRAY_TASK_ID"

dt=(0 0 0 0 1 1 1 1)
ss=(0.5 0.1 0.05 0.01 0.5 0.1 0.05 0.01)

julia ACE.jl 0 ${dt[$SLURM_ARRAY_TASK_ID]} ${ss[$SLURM_ARRAY_TASK_ID]}
julia ACE.jl 1 ${dt[$SLURM_ARRAY_TASK_ID]} ${ss[$SLURM_ARRAY_TASK_ID]}

