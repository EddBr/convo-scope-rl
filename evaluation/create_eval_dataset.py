import datasets

dataset = datasets.load_from_disk("/home/s2289391/convo-plan-SCOPE/lmsys_chat_1m_filtered")["train"]

l = len(dataset)
dataset_1 = dataset.select(range(l-1, l))
dataset_2 = dataset.select(range(l-2, l))
dataset_10 = dataset.select(range(l-10, l))
dataset_100 = dataset.select(range(l-100, l))
dataset_1000 = dataset.select(range(l-1000, l))


dataset_1.save_to_disk("eval_dataset_1")
dataset_2.save_to_disk("eval_dataset_2")
dataset_10.save_to_disk("eval_dataset_10")
dataset_100.save_to_disk("eval_dataset_100")
dataset_1000.save_to_disk("eval_dataset_1000")
