### Problem (train_bpe): BPE Tokenizer Training (15 points)

Full code is ```class BPETrainer```, which can be foune in ```./Tokenizer.py```. The results of testing are below:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest tests/test_train_bpe.py
================================================================================== test session starts ==================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 3 items                                                                                                                                                                       

tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

=================================================================================== 3 passed in 2.81s ===================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Wed 20 Aug 2025 07:05:57 PM CST
```

### Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)

Full code is ```./train_bpe_tinystories.py```. The results are below:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics/cs336_basics/Token$ uv run scalene train_bpe_tinystories.py 
Training BPE on TinyStoriesV2-GPT4...
Finished training.

Saving the "vocabulary" and "merges" to TinyStoriesV2-GPT4...
Saved.

This is the longest tokens in the vocabulary:
[(7183, b' accomplishment'), (8990, b' disappointment'), (9200, b' responsibility'), (3245, b' uncomfortable'), (3538, b' compassionate')]

(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics/cs336_basics/Token$ date
Thu 21 Aug 2025 06:56:08 PM CST
```
and
```./profile_train_bpe_tinystories.html```

### Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)

Full code is ```./train_bpe_expts_owt.py```. The results are below:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics/cs336_basics/Token$ uv run scalene train_bpe_expts_owt.py
Training BPE on OpenWebText...
Finished training.

Saving the "vocabulary" and "merges" to OpenWebText...
Saved.

This is the longest tokens in the vocabulary:
[(25822, b'\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82\xc3\x83\xc3\x82'), (25836, b'----------------------------------------------------------------'), (31274, b'\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94\xe2\x80\x94'), (10900, b'--------------------------------'), (15947, b'________________________________')]

(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics/cs336_basics/Token$ date
Fri 22 Aug 2025 11:26:08 AM CST
```
and
```./profile_train_bpe_expts_owt.html```