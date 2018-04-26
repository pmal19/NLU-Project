#!/bin/bash


#SBATCH --verbose
#SBATCH --job-name=quoraTrainer
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=50GB
###SBATCH --partition=gpu
<<<<<<< HEAD
#SBATCH --gres=gpu:p1080:1
#SBATCH --output=out.quoraTrainer.%j
=======
#SBATCH --gres=gpu:1
#SBATCH --output=out.snliTrainer.%j
>>>>>>> 69551a8a869634b5d6e92b66b51d1decaf2613ec

module load pytorch/python2.7/0.3.0_4
#module load pytorch/python3.6/0.3.0_4
python ./quoraEncoder.py
