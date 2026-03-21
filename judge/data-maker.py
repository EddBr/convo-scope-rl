print("importing libs")
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from litellm import completion
import os
import re
from openai import OpenAI

client = OpenAI(base_url="http://landonia11.inf.ed.ac.uk:8000/v1", api_key="token")

print("loading dataset")
dataset = load_from_disk("/home/s2289391/convo-plan-SCOPE/lmsys_chat_1m_filtered")["train"]
print("loaded dataset")

start = 0
end = 130_000


embeddings = []
for i, conversation in zip(tqdm(range(start, end)), iter(dataset)):
    e = []
    conversation = conversation['conversation']
    for j in range(1,len(conversation),2):
        try:
            # Here is where you make a request to the LLM
            c = conversation[:j+1] # this does up to j
            llm_resp = client.chat.completions.create(
                    model="llama-3.1-8b-instruct",
                    messages = [{"role":"system","content":"""### Role
You are an impartial judge evaluating the quality of an AI assistant's response.

### Conversation History
                                 """+str(c[:j])+"""

### Current Turn
User: """+str(c[j-1]["content"])+"""
Assistant: """+str(c[j]["content"])+"""

### Instructions
Output your response as a singular number between 1 and 5

### Evaluation Criteria
1. Helpful: Does it solve the user's problem?
2. Conciseness: Is it free of unnecessary 'filler' text?

                                 Score:"""}],
    max_tokens=1,
    temperature=0,
    logit_bias= {
        "16": 10,#1
        "17": 10,#2
        "18": 10,#3
        "19": 10,#4
        "20": 10,#5
        }
  )
            #score = re.search(r"[1-5]",llm_resp.choices[0].message.content)
            score = llm_resp.choices[0].message.content
            if score:
                #embeddings.append(score.group())
                embeddings.append(score)
            else:
                print("couldn't find score")
                print(llm_resp.choices[0].message.content)
                embeddings.append(3.5)

        except Exception as g:
            print("Failed to judge!")
            print(g)
            break

dataset = Dataset.from_dict(
        {
            "judgements": embeddings
            }
        )

os.makedirs(f"judgements", exist_ok=True)
dataset.save_to_disk(f"judgements/lmsys-{end}")
