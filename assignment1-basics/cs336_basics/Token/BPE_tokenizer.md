完整代码在 ```./Tokenizer.py``` 中的 ```class BPETrainer```, 测试结果如下:
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