#!/bin/bash

#SBATCH --job-name=NLI
#SBATCH -t72:00:00
#SBATCH --mem=100GB
#SBATCH --output=out.%j

module load pytorch/python2.7/0.3.0_4
#module load pytorch/python3.6/0.3.0_4
python ./quoraEncoder.py
