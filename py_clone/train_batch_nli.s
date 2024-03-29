#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=NLItrain
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
##SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=out.NLItrain.%j
source activate pmm/bin/activate
module load pytorch/python2.7/0.3.0_4
#module load pytorch/python3.6/0.3.0_4
/home/pm2758/pmm/bin/python ./train_batch_nli.py
