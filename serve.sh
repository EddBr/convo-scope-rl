#!/bin/sh
source ../convo-plan-SCOPE/venv/bin/activate
python3.12 train/stupid_embed.py
#python3.12 train/embed_dataset.py
#python3.12 train/train_transition.py --batch_size 1024 --dataset embeddings/lmsys-chat-1m_embeddings_1024_10000
