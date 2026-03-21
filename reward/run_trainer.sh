#!/bin/sh
sbatch -p Teaching --nodelist=landonia11 --gres=gpu:nvidia_rtx_a6000:1 go_trainer.sh
