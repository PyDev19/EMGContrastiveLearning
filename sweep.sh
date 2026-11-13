#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=h200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=0-7
#SBATCH --mail-user=axm240143@utdallas.edu
#SBATCH --mail-type=FAIL,ARRAY_TASKS
#SBATCH --job-name=bep_sweep
#SBATCH --output=./slurm_logs/sweep_%A_%a/log.out
#SBATCH --error=./slurm_logs/sweep_%A_%a/log.err

module load miniconda
source activate base
conda activate /groups/emeyers/.conda/envs/research/

SRC=/groups/emeyers/EMGContrastiveLearning/
cd "$SRC" || exit 1

export LD_LIBRARY_PATH=/opt/ohpc/pub/compiler/gcc/14.2.0/lib64:$LD_LIBRARY_PATH
export WANDB_DIR=/groups/emeyers/wandb_tmp
mkdir -p "$WANDB_DIR"

sleep $((SLURM_ARRAY_TASK_ID * 10))

wandb agent --count 1 "$1"
