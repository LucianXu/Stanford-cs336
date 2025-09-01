import numpy as np
import numpy.typing as npt

from tqdm import tqdm
from cs336_basics.Tokenizer.Tokenizer import Tokenizer

print("Initializing BPE-tokenizer...")
tokenizer = Tokenizer.from_files(vocab_filepath=r"../Tokenizer/TinyStoriesV2-GPT4/vocab.yaml", 
                                 merges_filepath=r"../Tokenizer/TinyStoriesV2-GPT4/merges.yaml", 
                                 special_tokens=["<|endoftext|>"])
print("Finished initializing BPE-tokenizer.\n")

def GetEncodedData(file_path: str) -> list[npt.NDArray]:
    text, data = [], []
    with open(file_path, "r") as f:
        text_temp = f.read()
        text_temp = text_temp.split("<|endoftext|>")
        for seq in text_temp:
            text.append(seq[:-1] + "<|endoftext|>")
    for seq in tqdm(text):
        data.append(tokenizer.encode(seq))
    return data

print("Saving encoded datas...")
train_data = GetEncodedData(r"../../data/TinyStoriesV2-GPT4-train.txt")
np.savez(r"../../data/TinyStoriesV2-GPT4-train.np", *train_data)

valid_data = GetEncodedData(r"../../data/TinyStoriesV2-GPT4-valid.txt")
np.savez(r"../../data/TinyStoriesV2-GPT4-valid.np", *valid_data)
print("Finished saving encoded datas...")