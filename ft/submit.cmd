#!/bin/bash
# parallel job using 20 processors. and runs for 1 hours (max) 
#SBATCH -N 1
#SBATCH --ntasks-per-node=7
#SBATCH -c 4
#SBATCH -t 30:00:00
#SBATCH --partition=parallel
##SBATCH -C ivy
##SBATCH --mem=80000
# sends mail when process begins, and
# when it ends. Make sure you define your email
# address.

#SBATCH --job-name=ft_rdm
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sunchong137@gmail.com

srun hostname |sort
ulimit -l unlimited
source /home/sunchong/.modulerc
export PYTHONPATH=/home/sunchong/work:$PYTHONPATH
#export OMP_NUM_THREADS=20
export SCRATCHDIR="/scratch/local/sunchong"
srun python ftfci.py > test_ft_6_U4.out

