from openai import OpenAI

from transformers import AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("/home/s2289391/llama-3.2-1b")

#for t in ["1", "2", "3", "4", "5", " 1", " 2", " 3", " 4", " 5"]:
#    print(repr(t), tokenizer.encode(t, add_special_tokens=False))

client = OpenAI(base_url="http://saxa.inf.ed.ac.uk:8000/v1", api_key="token")

response = client.chat.completions.create(
    model="llama-3.2-1b",
    messages=[{"role": "user", "content": "Rate this movie from 1 to 5. Explain why"}],
    max_tokens=1,
    temperature=0,
    logit_bias= {
        "16": 100,#1
        "17": 100,#2
        "18": 100,#3
        "19": 100,#4
        "20": 100,#5
        }
  )

print(response.choices[0].message.content)
