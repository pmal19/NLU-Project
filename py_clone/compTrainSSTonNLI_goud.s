#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=SSTonNLI
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
##SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out.SSTonNLI.%j
source activate hmm/bin/activate
module load pytorch/python2.7/0.3.0_4
#module load pytorch/python3.6/0.3.0_4
/home/sgm400/hmm/bin/python ./compTrainSSTonNLI.py
