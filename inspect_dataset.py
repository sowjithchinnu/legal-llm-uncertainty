from datasets import load_dataset

dataset = load_dataset("nguha/legalbench", "rule_qa")

print(dataset)
print()
print(dataset["test"][0])
