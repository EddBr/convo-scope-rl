#!/bin/sh

export TRANSFORMERS_OFFLINE=1
source ../convo-plan-SCOPE/venv/bin/activate
#python3.12 train/stupid_embed.py
#python3.12 train/embed_dataset.py
#python3.12 train/train_transition.py --batch_size 1024 --dataset embeddings/lmsys-chat-1m_embeddings_1024_10000
#. /home/htang2/toolchain-20251006/toolchain.rc
#nvcc --version
#pip install torch

#pip install flash-attn --no-build-isolation

#python3 -u evaluation/run_evaluation.py --reward_func=length_human --cuda_for_llm_reward=0 --cuda_for_q_embedding_transition=0 --lr=0.0001 --transition_model_dir=/home/s2289391/convo-scope-rl/transition_models/deterministic/seed_0_batch_1024 --evaluation_depth=4 --mcts_time=10 --agent=semantic_online --result_file=camera --trials=1 --evaluation_data=test_set_10.txt 2>&1 | tee output.out
python3 -u evaluation/run_evaluation.py --reward_func=judge --cuda_for_llm_reward=0 --cuda_for_q_embedding_transition=0 --lr=0.0001 --transition_model_dir=/home/s2289391/convo-scope-rl/transition_models/deterministic/seed_0_batch_1024 --evaluation_depth=4 --mcts_time=10 --agent=semantic_online --result_file=camera --trials=1 --evaluation_data=test_set_10.txt 2>&1 | tee output.out
#python3 -u evaluation/run_evaluation_singular.py --reward_func=judge --cuda_for_llm_reward=0 --cuda_for_q_embedding_transition=0 --lr=0.0001 --transition_model_dir=/home/s2289391/convo-scope-rl/transition_models/deterministic/seed_0_batch_1024 --evaluation_depth=4 --mcts_time=10 --agent=semantic_online --result_file=camera --trials=1 --evaluation_data=conversation_starter.txt 2>&1 | tee output.out
