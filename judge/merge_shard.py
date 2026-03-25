from datasets import concatenate_datasets, load_from_disk
import os

step = 13_000

pre = "/home/s2289391/convo-scope-rl/judge/shards/"
dss = []
for i in range(0,10):
    dss.append(f"{pre}j_{step*i}_{step*(i+1)}")

ds = concatenate_datasets([load_from_disk(shard) for shard in dss])


ds.save_to_disk("judgements/lmsys_130000_final")
