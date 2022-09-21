# Copyright (c) Meta Platforms, Inc. and affiliates
import distutils.command.clean
import glob
import os
import shutil
import subprocess
from typing import Dict, Union, List

from setuptools import setup, find_namespace_packages

# install: run `python spmd/setup.py install`

# Package name
package_name = "spmd"

# Version information
cwd: str = os.path.dirname(os.path.abspath(__file__))
version_txt: str = os.path.join(cwd, "version.txt")
with open(version_txt, "r") as f:
    version: str = f.readline().strip()

try:
    sha: str = (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
        .decode("ascii")
        .strip()
    )
except Exception:
    sha = "Unknown"

if os.getenv("BUILD_VERSION"):
    version = os.getenv("BUILD_VERSION", version)
elif sha != "Unknown":
    version += "+" + sha[:7]


def write_version_file() -> None:
    version_path = os.path.join(cwd, "version.py")
    with open(version_path, "w") as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))


# Package requirements
requirements = [
    # This represents a nightly version of PyTorch.
    # It can be installed as a binary or from source.
    "torch>=1.13.0.dev"
]

extras: Dict[str, Union[str, List[str]]] = {}


class clean(distutils.command.clean.clean):  # type: ignore
    def run(self) -> None:

        with open(".gitignore", "r") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)  # type: ignore


if __name__ == "__main__":
    write_version_file()

    setup(
        # Metadata
        name=package_name,
        version=version,
        author="SPMD Team",
        url="https://github.com/pytorch/PiPPy/spmd",
        description="SPMD implementation for PyTorch",
        license="BSD",
        # Package info
        packages=find_namespace_packages(),
        install_requires=requirements,
        extras_require=extras,
        cmdclass={"clean": clean},
    )
