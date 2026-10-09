#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=axm240143@utdallas.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=bep_train
#SBATCH --output=./slurm_logs/train_%j/log.out
#SBATCH --error=./slurm_logs/train_%j/log.err

module load miniconda

conda init bash
source activate base
conda activate /groups/emeyers/.conda/envs/meyerlab

SRC=/groups/emeyers/EMGContrastiveLearning/

cd $SRC

export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH

python -m src.train --data-dir $1 --config $2
