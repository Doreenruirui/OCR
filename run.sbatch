#!/bin/bash
#SBATCH --job-name=60000
#SBATCH --output=/home/dong.r/nlc-master/script/60000.out
#SBATCH --error=/home/dong.r/nlc-master/script/60000.err
#SBATCH --exclusive
#SBATCH --partition=par-gpu-2
#SBATCH -N 1

/home/dong.r/nlc-master/decode_line.sh 60000 80000
