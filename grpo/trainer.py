# train_grpo.py
import datasets
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

def num_turns_reward(completions):
    rewards = []
    for c in completions:
        rewards.append(len(c))
    return rewards


def assistant_len_reward(completions):
    rewards = []
    for c in completions:
        pass#TODO
    return rewards


def assistant_len_reward(completions):
    rewards = []
    for c in completions:
        pass#TODO
    return rewards

dataset = datasets.load_from_disk("/home/s2289391/convo-plan-SCOPE/lmsys_chat_1m_filtered")["train"]

trainer = GRPOTrainer(
    model="/home/s2289391/llama-3.1-8b-instruct",
    reward_funcs=[num_turns_reward],
    train_dataset=dataset,
)
trainer.train()
