#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --nodes=1         # number of nodes
#SBATCH --cpus-per-task=12
### SBATCH --mem-per-cpu=6G   # memory per CPU core
#SBATCH -J "barotropic_call"   # job name
#SBATCH -o "barotropic_calibration"

# Size of the ensemble
module load julia/1.7.1

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:=1}

echo $JULIA_NUM_THREADS

julia GMKI_Barotropic.jl |& tee gmki.log
