#!/bin/bash

#SBATCH --job-name=SST
#SBATCH -t72:00:00
#SBATCH --mem=100GB
#SBATCH --output=out.sst.%j

module load pytorch/python2.7/0.3.0_4
#module load pytorch/python3.6/0.3.0_4
python ./gumbelcompTrainer.py
