# pyre-ignore-all-errors
#
# util file to copy files to core, note that this file should
# be run from the tau/ root directory, and pytorch directory
# should be in the same directory as tau.
#
# cmd: python spmd/copy_to_core.py
#
import shutil
import os
import glob
import re

# path to source/dest directory
tau_src_dir = "./spmd/tensor/"
pytorch_dest_dir = "../pytorch/torch/distributed/_tensor/"

tau_test_dir = "./test/spmd/tensor/"
pytorch_test_dir = "../pytorch/test/distributed/_tensor/"

common_testing_dtensor_folder = [
    "./spmd/testing/common_dtensor.py",
    "./spmd/testing/dtensor_lagging_op_db.py",
    "./spmd/testing/gen_dtensor_lagging_op_db.py",
]
pytorch_common_testing_dtensor_folder = [
    "../pytorch/torch/testing/_internal/distributed/_tensor/common_dtensor.py",
    "../pytorch/torch/testing/_internal/distributed/_tensor/dtensor_lagging_op_db.py",
    "../pytorch/torch/testing/_internal/distributed/_tensor/gen_dtensor_lagging_op_db.py",
]

# rm folders if already exist, for copy_tree to work
if os.path.exists(pytorch_dest_dir):
    shutil.rmtree(pytorch_dest_dir)

if os.path.exists(pytorch_test_dir):
    shutil.rmtree(pytorch_test_dir)

# First, copying all the files in the source directory to pytorch directory
shutil.copytree(tau_src_dir, pytorch_dest_dir)
shutil.copytree(tau_test_dir, pytorch_test_dir)
for comm_test_file, pytorch_comm_test_file in zip(
    common_testing_dtensor_folder, pytorch_common_testing_dtensor_folder
):
    shutil.copy(comm_test_file, pytorch_comm_test_file)

# Second, we loop through all files for the two folder, regex replace
# imports, and write back to tbe original file
from_import_pattern = re.compile(r"from spmd.tensor")
replace_from_import_pattern = "from torch.distributed._tensor"
import_pattern = re.compile(r"import spmd.tensor")
replace_import_pattern = "import torch.distributed._tensor"

from_import_testing_pattern = re.compile(r"from spmd.testing")
replace_from_import_testing_pattern = "from torch.testing._internal.distributed._tensor"
import_testing_pattern = re.compile(r"import spmd.testing")
replace_import_testing_pattern = "import torch.testing._internal.distributed._tensor"


for filename in glob.iglob(pytorch_dest_dir + "**/*.py", recursive=True):
    with open(filename, "r") as f:
        code = f.read()
    # replace from import and import statements
    code = from_import_pattern.sub(replace_from_import_pattern, code)
    code = import_pattern.sub(replace_import_pattern, code)

    # Write the file out again
    with open(filename, "w") as f:
        f.write(code)

for filename in glob.iglob(pytorch_test_dir + "**/*.py", recursive=True):
    with open(filename, "r") as f:
        code = f.read()
    # replace from import and import statements
    code = from_import_pattern.sub(replace_from_import_pattern, code)
    code = import_pattern.sub(replace_import_pattern, code)
    code = from_import_testing_pattern.sub(
        replace_from_import_testing_pattern, code
    )
    code = import_testing_pattern.sub(replace_import_testing_pattern, code)

    # Write the file out again
    with open(filename, "w") as f:
        f.write(code)

# Last rewrite common_dtensor.py imports

for filename in pytorch_common_testing_dtensor_folder:
    with open(filename, "r") as f:
        code = f.read()

    # replace from import and import statements
    code = from_import_pattern.sub(replace_from_import_pattern, code)
    code = import_pattern.sub(replace_import_pattern, code)

    # Write the file out again
    with open(filename, "w") as f:
        f.write(code)
