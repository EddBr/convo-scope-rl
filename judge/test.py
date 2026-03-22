from datasets import load_from_disk, Dataset
#dataset = load_from_disk("/home/s2289391/convo-scope-rl/judge/judgements/lmsys-100")
#dataset = load_from_disk("/home/s2289391/convo-plan-SCOPE/lmsys_chat_1m_filtered")["train"]
#embeddings = load_from_disk("/home/s2289391/convo-scope-rl/embeddings/lmsys-chat-1m_embeddings_1024_130000").with_format("torch")
judgements = load_from_disk("/home/s2289391/convo-scope-rl/judge/judgements/lmsys-130000").with_format("torch")
print(len(judgements)) #130k
