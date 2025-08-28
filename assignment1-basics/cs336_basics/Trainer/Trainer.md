### Problem (linear): Implementing the linear module (1 point)

Full code is ```CrossEntropy```, which can be foune in ```./Loss.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_cross_entropy
================================================================================================ test session starts =================================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                                      

tests/test_nn_utils.py::test_cross_entropy PASSED

========================================================================================== 1 passed, 47 deselected in 0.07s ==========================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Wed 27 Aug 2025 06:47:05 PM CST
```