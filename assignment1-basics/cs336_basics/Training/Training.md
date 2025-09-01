### Problem (data_loading): Implement data loading (2 points)

Full code is ```GetBatch```, which can be foune in ```./DataLoader.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_get_batch
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collecting ... lr = 10.0, losses = [23.9643611907959, 15.337191581726074, 11.30592155456543, 8.84567642211914, 7.1649980545043945, 5.940604209899902, 5.0101118087768555, 4.281283378601074, 3.6972246170043945, 3.220694065093994]
lr = 100.0, losses = [23.9643611907959, 23.964357376098633, 4.11163330078125, 0.09840065240859985, 8.770762132766753e-17, 9.77555844060574e-19, 3.2917742949427234e-20, 1.960932818201212e-21, 1.6822147046008333e-22, 1.8691276599193895e-23]
lr = 1000.0, losses = [23.9643611907959, 8651.1337890625, 1494187.0, 166212336.0, 13463197696.0, 849682038784.0, 43619901767680.0, 1876714682580992.0, 6.917170243253043e+16, 2.221180064119128e+18]
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_data.py::test_get_batch PASSED

================================================================================= 1 passed, 47 deselected in 2.02s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Thu 28 Aug 2025 08:35:11 PM CST
```

### Problem (checkpointing): Implement model checkpointing (1 point)

Full code are ```SaveCheckpoint``` and ```LoadCheckpoint```, which can be foune in ```./Checkpoint.py```. The results of testing are as follows:
```sh
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ uv run pytest -k test_checkpointing
======================================================================================== test session starts ========================================================================================
platform linux -- Python 3.13.6, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/rxxu/my/stanford-cs336/assignment1-basics
configfile: pyproject.toml
plugins: jaxtyping-0.3.2
collecting ... lr = 10.0, losses = [25.222389221191406, 16.1423282623291, 11.899435043334961, 9.310037612915039, 7.541130542755127, 6.252460956573486, 5.273121356964111, 4.506032943725586, 3.8913137912750244, 3.3897669315338135]
lr = 100.0, losses = [25.222389221191406, 25.22238540649414, 4.327476978302002, 0.1035662591457367, 9.843688179141935e-17, 1.097140207844023e-18, 3.694456480008621e-20, 2.2008133783060645e-21, 1.8879998633449177e-22, 2.0977777311202515e-23]
lr = 1000.0, losses = [25.222389221191406, 9105.28125, 1572625.5, 174937792.0, 14169960448.0, 894286757888.0, 45909761064960.0, 1975234521464832.0, 7.2802916892672e+16, 2.337782722488959e+18]
collected 48 items / 47 deselected / 1 selected                                                                                                                                                     

tests/test_serialization.py::test_checkpointing PASSED

================================================================================= 1 passed, 47 deselected in 0.97s ==================================================================================
(cs336) rxxu@BNU:~/my/stanford-cs336/assignment1-basics$ date
Thu 28 Aug 2025 08:45:56 PM CST
```

### Problem (training_together): Put it together (4 points)
Skipped