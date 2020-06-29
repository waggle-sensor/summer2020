#!/bin/bash
#SBATCH --jobn-name=train_chars
#SBATCH -N 2
#SBATCH --gpu=4
#SBATCH --output=train_chars.out
#SBATCH --error=train_chars.error
#SBATCH --mail-user=sng@anl.gov
#SBATCH --time=05:00:00

# For blues only
# module load python/3.6.5-lvrzbky

# For lambda only
# . /software/anaconda3/bin/activate train

conda activate train
python3 train.py --data_config config/chars.data