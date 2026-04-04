print("importing libs")
#from datasets import load_from_disk, Dataset
from litellm import completion
import os
import re
from openai import OpenAI
import sys
import json

file_name = "/home/s2289391/convo-scope-rl/single_results_semantic_online_RESOLUTION_FINAL_10s_100.json" #SCOPE
#file_name = "/home/s2289391/convo-scope-rl/single_results_zero_step_greedy_RESOLUTION_FINAL_0s_100.json" #BASELINE
client = OpenAI(base_url="http://landonia11.inf.ed.ac.uk:8000/v1", api_key="token")
print("loading dataset")
with open(file_name) as f:
    dataset = json.load(f)
#dataset = load_from_disk("/home/s2289391/convo-plan-SCOPE/evaluation/single_results_eval_dataset_100_10" chooseit!)["train"]
print("loaded dataset")

#start = int(sys.argv[1])
#end = int(sys.argv[2])
#start=100

#dataset = dataset.select(range(start, end))

embeddings = []
for i in range(len(dataset)):
                           #for i, conversation in zip(tqdm(range(start, end)), iter(dataset)):
    try:
        # Here is where you make a request to the LLM
        llm_resp = client.chat.completions.create(
                model="llama-3.1-8b-instruct",
                messages = [{"role":"system","content":"""### Role
You are an impartial judge evaluating the quality of an AI assistant's response.

### Current Turn
User: """+(dataset[i]["starter"])+"""
Assistant: """+(dataset[i]["response"])+"""

### Instructions
Output your response as a singular number between 1 and 5

### Evaluation Criteria
1. Helpful: Does it solve the user's problem?

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
            try:
                embeddings.append(float(score))
            except:
                print("Fail!")
                embeddings.append(3)
        else:
            print("couldn't find score")
            print(llm_resp.choices[0].message.content)
            embeddings.append(3)

    except Exception as g:
        print("Failed to judge!")
        print("i",i)
        print("j",j)
        print("conversation",conversation)
        print(g)
        #Need to do this to account for missing j
        total_turns = len(range(1,len(conversation),2))
        completed =  (j-1) // 2
        missing = total_turns - completed
        for x in range(missing):
            embeddings.append(3.5)
        break

dd = {"data":embeddings}
import json
with open(f"single_results_RESOLUTION_FINAL_100_SCOPE_SCORES.json", "w") as f:
    json.dump(dd, f)
#os.makedirs("batch", exist_ok=True)
#dataset.save_to_disk(f"code_eval/end_SCOPE")


#os.makedirs(f"judgements", exist_ok=True)
#dataset.save_to_disk(f"judgements/lmsys-{end}")
