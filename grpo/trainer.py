# train_grpo.py
import datasets
from trl import GRPOTrainer
from trl.rewards import accuracy_reward

dataset = datasets.load_from_disk("/afs/inf.ed.ac.uk/user/s22/s2289391/convo-plan-SCOPE/lmsys_chat_1m_local_no_split")["train"]

trainer = GRPOTrainer(
    model="/afs/inf.ed.ac.uk/user/s22/s2289391/convo-plan-SCOPE/llama-3.2-1b-instruct",
    reward_funcs=accuracy_reward,
    train_dataset=dataset,
)
trainer.train()
