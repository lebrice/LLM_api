#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=72:00:00                                   # The job will run for 3 days
#SBATCH -o /network/scratch/<u>/<username>/slurm-%j.out  # Write the log on scratch

#  TODO: Write an sbatch script to start the server with a given model capacity.
export MODEL_CAPACITY="13b"
export PORT=12345
export TRANSFORMERS_CACHE=~/scratch
uvicorn app.server:app --port $PORT
