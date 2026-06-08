#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=axm240143@utdallas.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=transformer_cls_train
#SBATCH --output=transformer_cls_train.out
#SBATCH --nodelist=g-07-04

module load miniconda

conda init bash
source activate base
conda activate /groups/emeyers/.conda/envs/meyerlab

SRC=/groups/emeyers/EMGContrastiveLearning/

cd ~/scratch/physiomio/
DATA_DIR=$(pwd)/all_preprocessed/raw_ungrouped_labels

LOGS_DIR=$SRC/logs/transformer_cls/
CONFIG=$SRC/configs/transformer_cls_tfc.json
FOLD=1

cd $SRC

python -m src.train --data_path $DATA_DIR --config $CONFIG --log_dir $LOGS_DIR --fold $FOLD