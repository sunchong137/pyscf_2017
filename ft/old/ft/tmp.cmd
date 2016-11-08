#!/bin/bash
#SBATCH -A chem
#SBATCH -C ivy
#SBATCH -N 1   # node count
#SBATCH -t 1:00:00
#SBATCH --cpus-per-task=1
#export OMP_NUM_THREADS=1
python hs.py 
