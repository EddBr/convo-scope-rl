#!/bin/sh
sbatch -p Teaching --nodelist=saxa --gres=gpu:1g.18gb:1 grpo_serve.sh
