## Run distributed tensor op db tests:

from root, run (either CPU or GPU)

`pytest test/spmd/tensor/test_dtensor_ops.py`

run specific test case and print stdout/stderr:

`pytest test/spmd/tensor/test_dtensor_ops.py -s -k addmm`
