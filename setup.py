# Copyright (c) Meta Platforms, Inc. and affiliates
import distutils.command.clean
import os
import subprocess
from typing import Dict

from setuptools import setup, find_packages

# Package name
package_name = 'pippy'

# Version information
cwd = os.path.dirname(os.path.abspath(__file__))
version_txt = os.path.join(cwd, 'version.txt')
with open(version_txt, 'r') as f:
    version = f.readline().strip()

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
except Exception:
    sha = 'Unknown'

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION', version)
elif sha != 'Unknown':
    version += '+' + sha[:7]

def write_version_file():
    version_path = os.path.join(cwd, 'pippy', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))
        f.write("git_version = {}\n".format(repr(sha)))

# Package requirements
requirements = [
    # This represents a nightly version of PyTorch.
    # It can be installed as a binary or from source.
    "torch>=1.10.0.dev",
]

extras: Dict = {}

class clean(distutils.command.clean.clean):  # type: ignore
    def run(self):

        with open(".gitignore", "r") as f:
            ignores = f.read()
            for wildcard in filter(None, ignores.split("\n")):
                for filename in glob.glob(wildcard):
                    try:
                        os.remove(filename)
                    except OSError:
                        shutil.rmtree(filename, ignore_errors=True)

        # It's an old-style class in Python 2.7...
        distutils.command.clean.clean.run(self)

if __name__ == '__main__':
    write_version_file()

    setup(
        # Metadata
        name=package_name,
        version=version,
        author='PiPPy Team',
        url="https://github.com/pytorch/PiPPy",
        description='Pipeline Parallelism for PyTorch',
        license='BSD',

        # Package info
        packages=find_packages(),
        install_requires=requirements,
        extras_require=extras,
        cmdclass={
            'clean': clean,
        })
