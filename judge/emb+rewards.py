from datasets import load_from_disk, Dataset
from tqdm import tqdm
import torch.nn as nn

start = 0
end = 130_000

embeddings = load_from_disk(f"/home/s2289391/convo-scope-rl/embeddings/lmsys-chat-1m_embeddings_1024_{end}").with_format("torch")
judge_rewards = load_from_disk(f"/home/s2289391/convo-scope-rl/judge/judgements/lmsys-{end}").with_format("torch")

e = []
r = []
counter = 0

for i, embedding in zip(tqdm(range(start, end)), iter(embeddings)):
    embedding = embedding['embeddings']
    z = 0
    for j in range(1,len(embedding),2):
        e.append(embedding[j])
        r.append(judge_rewards["judgements"][counter+z])
        z += 1

    counter += z

dataset_dict = {
        "embedding": e,
        "reward": r
        }

ds = Dataset.from_dict(dataset_dict)

ds.save_to_disk(f"embedding_rewards/lmsys_embeddings_rewards_{end}")
print("created ds")



#This is an old version I was trying
def create_reward_data(batch):
    conversations = batch["embeddings"]
    inputs = []
    targets = []
    for convo in conversations:
        n = len(convo)
        for c in range(1,n,2):
            inputs.append(convo[c])
            targets.append(n - c - 1)

    return {
            "inputs":inputs,
            "targets":targets
            }

#reward_ds = embeddings.map(
        #create_reward_data,
        #batched=True,
        #remove_columns=embeddings.column_names,
        #num_proc=16
        #)
