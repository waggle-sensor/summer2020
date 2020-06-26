#!/bin/bash
#SBATCH --jobname=augment_chars
#SBATCH -N 2
#SBATCH -gres=gpu:3
#SBATCH --output=augment_chars.out
#SBATCH --error=augment_chars.error
#SBATCH --mail-user=sng@anl.gov
#SBATCH --time=05:00:00

. /software/anaconda3/bin/activate
conda activate train
python3 train.py --data_config config/chars.data