#!/bin/sh
sbatch -p Teaching  --time=1-12:00:00 --nodelist=saxa --gres=gpu:3g.71gb:1 llm-server.sh #--nodelist=landonia11
#sbatch -p Teaching  --time=1-12:00:00 --nodelist=saxa --gres=gpu:1g.18gb:1 llm-server.sh #--nodelist=landonia11
