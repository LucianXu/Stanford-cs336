### Problem (linear): Implementing the linear module (1 point)

Full code is ```class Linear```, which can be foune in ```./Modules.py```. The results of testing are as follows:
``` sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_linear
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_model.py::test_linear PASSED

================================================================================= 1 passed, 47 deselected in 0.87s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Mon 25 Aug 2025 01:33:04 PM CST
```