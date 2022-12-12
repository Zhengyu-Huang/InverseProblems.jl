#!/bin/csh -f
#SBATCH --ntasks=32
##SBATCH --nodes=1
#SBATCH --mem-per-cpu=8G 
#SBATCH --time=2:00:00 
#SBATCH --job-name=fms_default
##SBATCH --output=out_err/slurm_%j.out
##SBATCH --error=out_err/slurm_%j.err 
#SBATCH -o output/slurm/array-%A_%a.out                                            
#SBATCH -e output/slurm/array-%A_%a.err                                              

#SBATCH --export=ALL
#SBATCH --partition=any






set run_num = 1

rm -rf /central/scratch/dzhuang/fms_$run_num

set SLURM_SUBMIT_DIR = /groups/esm/dzhuang/fms_GCMForcing/exp/held_suarez/run_$run_num/

source /groups/esm/dzhuang/fms_GCMForcing/exp/held_suarez/run_$run_num/run_control_$run_num
