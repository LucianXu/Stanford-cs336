from cs336_basics.Token.Tokenizer import BPETrainer

trainer = BPETrainer()
vocab, merges = trainer.train_BPE(
    input_path = r"/home/rxxu/my/stanford-cs336/assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt",
    vocab_size = 10000, 
    special_tokens = ["<|endoftext|>"])
