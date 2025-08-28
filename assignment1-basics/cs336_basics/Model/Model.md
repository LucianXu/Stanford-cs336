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

### Problem (embedding): Implement the embedding module (1 point)

Full code is ```class Embedding```, which can be foune in ```./Modules.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_embedding
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_model.py::test_embedding PASSED

================================================================================= 1 passed, 47 deselected in 0.09s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Tue 26 Aug 2025 02:33:19 PM CST
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ 
```

### Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)

Full code is ```class RootMeanSquareLayerNorm```, which can be foune in ```./Modules.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_rmsnorm
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_model.py::test_rmsnorm PASSED

================================================================================= 1 passed, 47 deselected in 0.07s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Tue 26 Aug 2025 02:54:43 PM CST
```

### Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)

Full code is ```class SwiGLUFeedFowardNeuralNerwork```, which can be foune in ```./Modules.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_swiglu
================================================================================================ test session starts =================================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                                      

tests/test_model.py::test_swiglu PASSED

========================================================================================== 1 passed, 47 deselected in 0.86s ==========================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Tue 26 Aug 2025 03:34:31 PM CST
```

### Problem (rope): Implement RoPE (2 points)

Full code is ```class RotaryPositionalEmbedding```, which can be foune in ```./Modules.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_rope
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_model.py::test_rope PASSED

================================================================================= 1 passed, 47 deselected in 0.06s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Tue 26 Aug 2025 05:19:47 PM CST
```

### Problem (softmax): Implement softmax (1 point)

Full code is ```softmax```, which can be foune in ```./Modules.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_softmax
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED

================================================================================= 1 passed, 47 deselected in 0.07s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_softmax_matches_pytorch
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_nn_utils.py::test_softmax_matches_pytorch PASSED

================================================================================= 1 passed, 47 deselected in 0.11s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Tue 26 Aug 2025 08:56:31 PM CST
```

### Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)

Full code is ```class ScaledDotProductAttention```, which can be foune in ```./Modules.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_scaled_dot_product_attention
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_model.py::test_scaled_dot_product_attention PASSED

================================================================================= 1 passed, 47 deselected in 0.85s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_4d_scaled_dot_product_attention
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_model.py::test_4d_scaled_dot_product_attention PASSED

================================================================================= 1 passed, 47 deselected in 0.87s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Tue 26 Aug 2025 09:16:25 PM CST
```

### Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)

Full code is ```class CausalMultiHeadSelfAttention```, which can be foune in ```./Modules.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_multihead_self_attention
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 46 deselected / 2 selected                                                                                                                                                     

tests/test_model.py::test_multihead_self_attention PASSED
tests/test_model.py::test_multihead_self_attention_with_rope PASSED

================================================================================= 2 passed, 46 deselected in 0.88s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Tue 26 Aug 2025 10:46:50 PM CST
```

### Problem (transformer_block): Implement the Transformer block (3 points)

Full code is ```class TransformerBlock```, which can be foune in ```./Model.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_transformer_block
================================================================================================ test session starts =================================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 47 deselected / 1 selected                                                                                                                                                                      

tests/test_model.py::test_transformer_block PASSED

========================================================================================== 1 passed, 47 deselected in 0.98s ==========================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Wed 27 Aug 2025 04:38:55 PM CST
```

### Problem (transformer_lm): Implementing the Transformer LM (3 points)

Full code is ```class Transformer```, which can be foune in ```./Model.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_transformer_lm
================================================================================================ test session starts =================================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collected 48 items / 46 deselected / 2 selected                                                                                                                                                                      

tests/test_model.py::test_transformer_lm PASSED
tests/test_model.py::test_transformer_lm_truncated_input PASSED

========================================================================================== 2 passed, 46 deselected in 0.95s ==========================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Wed 27 Aug 2025 04:40:31 PM CST
```

### Problem (transformer_accounting): Transformer LM resource accounting (5 points)

Skipped.