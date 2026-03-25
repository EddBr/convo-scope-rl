#!/bin/sh
#SBATCH --job-name=judge_batched
#SBATCH --array=0-9
#SBATCH --ntasks=1
#
STEP=13000
START=$((SLURM_ARRAY_TASK_ID*STEP))
END=$((START+STEP))

python3 -u data-maker.py $START $END
