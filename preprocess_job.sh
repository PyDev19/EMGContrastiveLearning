#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=normal
#SBATCH --mail-user=axm240143@utdallas.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=project-blueprint-preprocess
#SBATCH --output=preprocess_output.txt

module load miniconda

conda init bash
source activate base
conda activate meyerlab

SRC=/groups/emeyers/EMGContrastiveLearning

cd ~/scratch/

hf download formove-ai/physiomio --repo-type dataset --local-dir ./blueprint_data/

python $SRC/src/preprocessing/physiomio.py --data_dir ./blueprint_data/data/ --output_dir ./blueprint_data/ --include_fma_zero
