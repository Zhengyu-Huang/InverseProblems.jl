#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=16G
#SBATCH -J "Gaussian"    # job name
#SBATCH -o "Gaussian"


# number of tasks

module purge
module load julia/1.8.1 


julia  Gaussian.jl 