import datasets
from trl import GRPOTrainer, PPOTrainer, PPOConfig,
from trl.rewards import accuracy_reward
from transformers import AutoModelForCausalLM, AutoTokenizer
#from peft import LoraConfig

model_dir = "/home/s2289391/llama-3.2-1b"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_dir)

dataset = datasets.load_from_disk("/home/s2289391/convo-plan-SCOPE/lmsys_chat_1m_filtered")["train"]

def num_turns_reward(completions):
    rewards = []
    for c in completions:
        rewards.append(-len(c))
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

dataset = dataset.select(range(10))


#lora_config = LoraConfig()
#model = get_peft_model(model,lora_config)


#Multi-turn step

trainer = PPOTrainer(
    model=model,
    reward_funcs=[num_turns_reward],
    train_dataset=dataset,
)
trainer.train()
