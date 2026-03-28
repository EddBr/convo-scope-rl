#!/bin/sh

#Single-turn
#sbatch -p Teaching --nodelist=saxa --gres=gpu:3g.71gb:1 serve_single_turn.sh
sbatch -p Teaching --nodelist=landonia11 --gres=gpu:nvidia_rtx_a6000:1 serve_single_turn.sh

#Multi-turn
#sbatch -p Teaching --nodelist=saxa --gres=gpu:1g.18gb:1 serve_multi_turn.sh
#sbatch -p Teaching --nodelist=saxa --gres=gpu:3g.71gb:1 serve_multi_turn.sh
#sbatch -p Teaching --nodelist=landonia11 --gres=gpu:nvidia_rtx_a6000:1 serve_multi_turn.sh

#--dependency=afterok:2226476
