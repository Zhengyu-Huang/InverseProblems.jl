#!/bin/bash

#SBATCH --time=168:00:00   # walltime
#SBATCH --nodes=1          # number of nodes
#SBATCH --ntasks=1         # number of processor cores (i.e. tasks)
#SBATCH --mem-per-cpu=128G
#SBATCH -J "Darcy-GF"      # job name
#SBATCH -o "Darcy-GF"




julia  Darcy-GF.jl 