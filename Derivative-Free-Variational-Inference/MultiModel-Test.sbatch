#!/bin/bash

#SBATCH --time=96:00:00   # walltime
#SBATCH --ntasks=12       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH -J "MultiModel-Test"   # job name
#SBATCH -o "MultiModel-Test"

# Size of the ensemble
# module load julia/1.7.1

export JULIA_NUM_THREADS=12 #${SLURM_CPUS_PER_TASK:=1}

echo $JULIA_NUM_THREADS

julia MultiModal-Test.jl |& tee MultiModal-Test.log
