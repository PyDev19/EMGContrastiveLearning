#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=axm240143@utdallas.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=bep_train
#SBATCH --output=bep_train.out

module load miniconda

conda init bash
source activate base
conda activate /groups/emeyers/.conda/envs/meyerlab

SRC=/groups/emeyers/EMGContrastiveLearning/

cd ~/scratch/blueprint_data/
DATA_DIR=$(pwd)/impaired_arm_ungrouped_include_fma_zero

CONFIG=$SRC/configs/roformer_contrastive_physiomio.yaml

cd $SRC

python -m src.train --data-dir $DATA_DIR --config $CONFIG