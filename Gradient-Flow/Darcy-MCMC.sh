#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1         # number of nodes
#SBATCH --ntasks=1       # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=128G
#SBATCH -J "Darcy-MCMC"    # job name
#SBATCH -o "Darcy-MCMC"


# number of tasks

module purge

julia  Darcy-MCMC.jl 