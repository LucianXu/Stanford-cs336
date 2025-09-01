import base64, yaml
from itertools import islice
from cs336_basics.Tokenizer.Tokenizer import BPETrainer

"""
Hoping you are under the directory: "stanford-cs336/assignment1-basics/cs336_basics/Token"
We will:
    1. Train a BPE tokenizer on the OpenWebText dataset.
    2. Save the vocabulary and merges to a specified directory.
    3. Take a look at the longest token in the vocabulary.
"""

# Initialize and train the BPE Trainer
trainer = BPETrainer()
print("Training BPE on OpenWebText...")
vocab, merges = trainer.train_BPE(
    input_path = r"../../data/owt_train.txt",
    vocab_size = 32000, 
    special_tokens = ["<|endoftext|>"])
print("Finished training.\n")

# Save and load functions for vocabulary and merges
def save(vocab, merges, path: str):
    vocab_encoded = {k: base64.b64encode(v).decode('ascii') if isinstance(v, bytes) else v for k, v in vocab.items()}
    merges_encoded = [(base64.b64encode(x).decode('ascii'), base64.b64encode(y).decode('ascii')) for x, y in merges]
    with open(f"{path}/vocab.yaml", 'w', encoding="utf-8") as f:
        yaml.dump(vocab_encoded, f, allow_unicode=True)
    with open(f"{path}/merges.yaml", 'w', encoding="utf-8") as f:
        yaml.dump(merges_encoded, f, allow_unicode=True)

def load(path: str):
    with open(f"{path}/vocab.yaml", 'r', encoding='utf-8') as f:
        vocab_encoded = yaml.safe_load(f)
    with open(f"{path}/merges.yaml", 'r', encoding='utf-8') as f:
        merges_encoded = yaml.load(f, Loader=yaml.FullLoader)
    vocab = {int(k): base64.b64decode(v) if isinstance(v, str) else v for k, v in vocab_encoded.items()}
    merges = [(base64.b64decode(x), base64.b64decode(y)) for x, y in merges_encoded]
    return vocab, merges

# Save the vocabulary and merges
print("Saving the \"vocabulary\" and \"merges\" to OpenWebText...")
save(vocab, merges, path="./OpenWebText")
print("Saved.\n")

# Print the longest token in the vocabulary
print("This is the longest tokens in the vocabulary:")
vocab_sorted = sorted(vocab.items(), key=lambda x: len(x[1]), reverse=True)
print(vocab_sorted[:5])