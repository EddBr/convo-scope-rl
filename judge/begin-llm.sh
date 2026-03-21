#!/bin/sh
#sbatch -p Teaching  --time=1-12:00:00 --nodelist=saxa --gres=gpu:3g.71gb:1 llm-server.sh 
#sbatch -p Teaching  --time=1-12:00:00 --nodelist=saxa --gres=gpu:1g.18gb:1 llm-server.sh
sbatch -p Teaching  --time=1-12:00:00 --nodelist=landonia11 --gres=gpu:nvidia_rtx_a6000:1 llm-server.sh 
