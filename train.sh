#!/usr/bin/bash
#SBATCH --qos=cbmm
#SBATCH -p cbmm
#SBATCH --mail-type=ALL
#SBATCH --mail-user=galanti@mit.edu
#SBATCH --gres=gpu:1
#SBATCH --mem=10000
#SBATCH --output=output_cifar10.sh
#SBATCH --time=01-00:00
echo $CUDA_VISIBLE_DEVICES
python train.py