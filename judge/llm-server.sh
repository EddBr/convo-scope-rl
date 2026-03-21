#!/bin/sh

nvidia-smi

echo "Server running on node: $(hostname)"
source llmvenv/bin/activate
#python -m vllm.entrypoints.openai.api_server \
#  --model /home/s2289391/llama-3.1-8b \
#  --served-model-name llama-3.1-8b \
#  --port 8000
CUDA_VISIBLE_DEVICES=0

vllm serve /home/s2289391/llama-3.1-8b-instruct --port 8000 --served-model-name llama-3.1-8b-instruct
#vllm serve /home/s2289391/llama-3.2-1b --port 8000 --served-model-name llama-3.2-1b
#vllm serve /home/s2289391/llama-3.2-3b --port 8000 --served-model-name llama-3.2-3b



#--enable-auto-tool-choice --tool-call-parser
