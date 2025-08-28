### Problem (cross_entropy): Implement Cross entropy

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

### Problem (learning_rate_tuning): Tuning the learning rate (1 point)

Full code can be foune in ```./SGD_test.py```. The results of testing are as follows:
```sh
lr = 10.0, losses = [32.50678634643555, 20.804346084594727, 15.336073875427246, 11.99884033203125, 9.719060897827148, 8.05821418762207, 6.796035289764404, 5.807405471801758, 5.015151500701904, 4.3687543869018555]
lr = 100.0, losses = [32.50678634643555, 32.50678634643555, 5.577281475067139, 0.13347692787647247, 1.5741802922255752e-16, 1.7545216303479575e-18, 5.908090996250544e-20, 3.519490736009393e-21, 3.0192465962710223e-22, 3.354718089696289e-23]
lr = 1000.0, losses = [32.50678634643555, 11734.9501953125, 2026810.5, 225461024.0, 18262341632.0, 1152563019776.0, 59168811646976.0, 2545695567380480.0, 9.382889493050163e+16, 3.012950481908531e+18]
```
It can be observed that with a small learning rate (10.0), the loss decreases steadily; with a moderate rate (100.0), the loss converges extremely fast; but with a large rate (1000.0), the training diverges explosively.

### Problem (adamw): Implement AdamW (2 points)

Full code is ```class AdamW```, which can be foune in ```./Optimizer.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_adamw
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collecting ... lr = 10.0, losses = [20.942014694213867, 13.4028902053833, 9.880038261413574, 7.730073928833008, 6.261361122131348, 5.191385269165039, 4.378244876861572, 3.741335153579712, 3.2309370040893555, 2.8145053386688232]
lr = 100.0, losses = [20.942014694213867, 20.9420108795166, 3.593080997467041, 0.0859905257821083, 1.0741624497803768e-16, 1.1972206867592372e-18, 4.0314631093772224e-20, 2.4015704764797895e-21, 2.0602222091243413e-22, 2.2891359807486005e-23]
lr = 1000.0, losses = [20.942014694213867, 7560.06689453125, 1305742.5, 145249936.0, 11765242880.0, 742521569280.0, 38118627475456.0, 1640026416873472.0, 6.044787653004493e+16, 1.9410486906176143e+18]
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_optimizer.py::test_adamw PASSED

================================================================================= 1 passed, 47 deselected in 1.60s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Thu 28 Aug 2025 03:07:40 PM CST
```

### Problem (adamwAccounting): Resource accounting for training with AdamW (2 points)

skipped.

### Problem (learning_rate_schedule): Implement cosine learning rate schedule with warmup

Full code is ```LearningRateScheduler```, which can be foune in ```./Optimizer.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_get_lr_cosine_schedule
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collecting ... lr = 10.0, losses = [23.812456130981445, 15.239973068237305, 11.23425579071045, 8.789606094360352, 7.11958122253418, 5.902947902679443, 4.978353500366211, 4.254144668579102, 3.6737892627716064, 3.2002782821655273]
lr = 100.0, losses = [23.812456130981445, 23.812454223632812, 4.085570812225342, 0.09777691960334778, 7.040375068433194e-17, 7.846935660637954e-19, 2.6423391248321426e-20, 1.5740595773320766e-21, 1.3503300670482978e-22, 1.5003667411647754e-23]
lr = 1000.0, losses = [23.812456130981445, 8596.2958984375, 1484715.625, 165158752.0, 13377859584.0, 844296093696.0, 43343404859392.0, 1864818562695168.0, 6.873323351624909e+16, 2.207100680286503e+18]
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_optimizer.py::test_get_lr_cosine_schedule PASSED

================================================================================= 1 passed, 47 deselected in 0.86s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Thu 28 Aug 2025 03:31:17 PM CST
```

### Problem (gradient_clipping): Implement gradient clipping (1 point)

Full code is ```GraidentClipping```, which can be foune in ```./Optimizer.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_gradient_clipping
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collecting ... lr = 10.0, losses = [27.674036026000977, 17.711383819580078, 13.056075096130371, 10.214984893798828, 8.274138450622559, 6.860208034515381, 5.785675525665283, 4.944024562835693, 4.269554138183594, 3.7192559242248535]
lr = 100.0, losses = [27.674036026000977, 27.674034118652344, 4.748112678527832, 0.11363302171230316, 1.5186636352801602e-16, 1.6926450050113877e-18, 5.699731308531826e-20, 3.3953688115834734e-21, 2.912767034808504e-22, 3.2364075710305005e-23]
lr = 1000.0, losses = [27.674036026000977, 9990.3271484375, 1725486.5, 191941952.0, 15547297792.0, 981212594176.0, 50372253057024.0, 2167229625925632.0, 7.987945103844966e+16, 2.565018240845611e+18]
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_nn_utils.py::test_gradient_clipping PASSED

================================================================================= 1 passed, 47 deselected in 0.86s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Thu 28 Aug 2025 03:57:18 PM CST
```