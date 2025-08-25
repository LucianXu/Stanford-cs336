import yaml
from itertools import islice
from cs336_basics.Token.Tokenizer import BPETrainer

"""
Hoping you are under the directory: "stanford-cs336/assignment1-basics/cs336_basics/Token"
We will:
    1. Train a BPE tokenizer on the TinyStoriesV2-GPT4 dataset.
    2. Save the vocabulary and merges to a specified directory.
    3. Take a look at the longest token in the vocabulary.
"""

# Initialize and train the BPE Trainer
trainer = BPETrainer()
print("Training BPE on TinyStoriesV2-GPT4...")
vocab, merges = trainer.train_BPE(
    input_path = r"../../data/TinyStoriesV2-GPT4-train.txt",
    vocab_size = 10000, 
    special_tokens = ["<|endoftext|>"])
print("Finished training.\n")

# Save and load functions for vocabulary and merges
def save(vocab, merges, path: str):
    vocab_decoded = {k: v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v for k, v in vocab.items()}
    merges_decoded = [(x.decode("utf-8", errors="replace"), y.decode("utf-8", errors="replace")) for x, y in merges]
    with open(f"{path}/vocab.yaml", 'w', encoding="utf-8") as f:
        yaml.dump(vocab_decoded, f, allow_unicode=True)
    with open(f"{path}/merges.yaml", 'w', encoding="utf-8") as f:
        yaml.dump(merges_decoded, f, allow_unicode=True)

def load(path: str):
    with open(f"{path}/vocab.yaml", 'r', encoding='utf-8') as f:
        vocab_decoded = yaml.safe_load(f)
    with open(f"{path}/merges.yaml", 'r', encoding='utf-8') as f:
        merges_decoded = yaml.safe_load(f)
    vocab = {k: v.encode("utf-8") if isinstance(v, str) else v for k, v in vocab_decoded.items()}
    merges = [(x.encode("utf-8"), y.encode("utf-8")) for x, y in merges_decoded]
    return vocab, merges

# Save the vocabulary and merges
print("Saving the \"vocabulary\" and \"merges\" to TinyStoriesV2-GPT4...")
save(vocab, merges, path="./TinyStoriesV2-GPT4")
print("Saved.\n")

# Print the longest token in the vocabulary
print("This is the longest tokens in the vocabulary:")
vocab_sorted = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)
print(vocab_sorted[:5])