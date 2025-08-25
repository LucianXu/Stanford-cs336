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

### Problem (tokenizer): Implementing the tokenizer (15 points)
Full code is ```class Tokenizer```, which can be foune in ```./Tokenizer.py```. The results of testing are below:
```sh
============================================================================= 10 failed, 14 passed, 1 xfailed in 11.66s =============================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest tests/test_tokenizer.py
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 25 items                                                                                                                                                                                  

tests/test_tokenizer.py::test_roundtrip_empty PASSED
tests/test_tokenizer.py::test_empty_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_character PASSED
tests/test_tokenizer.py::test_single_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_single_unicode_character PASSED
tests/test_tokenizer.py::test_single_unicode_character_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_ascii_string PASSED
tests/test_tokenizer.py::test_ascii_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string PASSED
tests/test_tokenizer.py::test_unicode_string_matches_tiktoken PASSED
tests/test_tokenizer.py::test_roundtrip_unicode_string_with_special_tokens PASSED
tests/test_tokenizer.py::test_unicode_string_with_special_tokens_matches_tiktoken PASSED
tests/test_tokenizer.py::test_overlapping_special_tokens PASSED
tests/test_tokenizer.py::test_address_roundtrip PASSED
tests/test_tokenizer.py::test_address_matches_tiktoken PASSED
tests/test_tokenizer.py::test_german_roundtrip PASSED
tests/test_tokenizer.py::test_german_matches_tiktoken PASSED
tests/test_tokenizer.py::test_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_special_token_trailing_newlines PASSED
tests/test_tokenizer.py::test_encode_special_token_double_newline_non_whitespace PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_sample_roundtrip PASSED
tests/test_tokenizer.py::test_encode_iterable_tinystories_matches_tiktoken PASSED
tests/test_tokenizer.py::test_encode_iterable_memory_usage PASSED
tests/test_tokenizer.py::test_encode_memory_usage XFAIL (Tokenizer.encode is expected to take more memory than allotted (1MB).)

============================================================================ 24 passed, 1 xfailed in 30311.52s ============================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Mon 25 Aug 2025 12:01:20 PM CST
```