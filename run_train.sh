#!/bin/sh
sbatch -p Teaching  --nodelist=saxa --gres=gpu:1g.18gb:1 serve.sh
#--dependency=afterok:2226476
