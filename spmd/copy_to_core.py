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

# path to source directory
tau_src_dir = "./spmd/tensor/"
tau_test_dir = "./test/spmd/tensor/"
common_testing_dtensor = "./spmd/testing/common_dtensor.py"

# path to destination directory
pytorch_dest_dir = "../pytorch/torch/distributed/_tensor/"
pytorch_test_dir = "../pytorch/test/distributed/_tensor/"
pytorch_common_testing_dtensor = (
    "../pytorch/torch/testing/_internal/common_dtensor.py"
)

# rm folders if already exist, for copy_tree to work
if os.path.exists(pytorch_dest_dir):
    shutil.rmtree(pytorch_dest_dir)

if os.path.exists(pytorch_test_dir):
    shutil.rmtree(pytorch_test_dir)

# First, copying all the files in the source directory to pytorch directory
shutil.copy(common_testing_dtensor, pytorch_common_testing_dtensor)
shutil.copytree(tau_src_dir, pytorch_dest_dir)
shutil.copytree(tau_test_dir, pytorch_test_dir)

# Second, we loop through all files for the two folder, regex replace
# imports, and write back to tbe original file
from_import_pattern = re.compile(r"from spmd.tensor")
replace_from_import_pattern = "from torch.distributed._tensor"
import_pattern = re.compile(r"import spmd.tensor")
replace_import_pattern = "import torch.distributed._tensor"

from_import_testing_pattern = re.compile(r"from spmd.testing")
replace_from_import_testing_pattern = "from torch.testing._internal"
import_testing_pattern = re.compile(r"import spmd.testing")
replace_import_testing_pattern = "import torch.testing._internal"


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

with open(pytorch_common_testing_dtensor, "r") as f:
    code = f.read()

# replace from import and import statements
code = from_import_pattern.sub(replace_from_import_pattern, code)
code = import_pattern.sub(replace_import_pattern, code)

# Write the file out again
with open(pytorch_common_testing_dtensor, "w") as f:
    f.write(code)
