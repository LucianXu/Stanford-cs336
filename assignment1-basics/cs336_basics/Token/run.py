from cs336_basics.Token.Tokenizer import Tokenizer

# trainer = BPETrainer()
# vocab, merges = trainer.train_BPE(
#     input_path = r"/home/rxxu/my/stanford-cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
#     vocab_size = 10000, 
#     special_tokens = ["<|endoftext|>"])

tokenizer = Tokenizer.from_files(
    vocab_filepath = "/home/rxxu/my/stanford-cs336/assignment1-basics/cs336_basics/Token/TinyStoriesV2-GPT4/vocab.yaml",
    merges_filepath = "/home/rxxu/my/stanford-cs336/assignment1-basics/cs336_basics/Token/TinyStoriesV2-GPT4/merges.yaml",
    special_tokens = ["<|endoftext|>", "<|endoftext|><|endoftext|>"]
)

text = "This is a text.<|endoftext|>This is a text, too.<|endoftext|><|endoftext|>"
encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
print(encoded, decoded)

