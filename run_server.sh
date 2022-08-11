#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=72:00:00                                   # The job will run for 3 days

module load miniconda/3
conda env create -p $SLURM_TMPDIR/env -f env.yaml
conda activate $SLURM_TMPDIR/env
python app/server.py $@
