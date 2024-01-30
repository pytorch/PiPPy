#! /usr/env/bin
import argparse
import glob
import os
import sys
import subprocess


"""
Instructions:

Make sure you have installed the following packages before running this script:
`pip install twine`

Make sure you have cleaned and then built wheel files locally:
see instructions in `build.py`

To upload to pypi, run:
`python upload.py --upload`
Then copy and paste the pypi token when prompted.
"""


# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)


def exe_cmd(cmd, dry_run=True):
    if dry_run:
        print(f"Executing command: {cmd}")
    else:
        try:
            subprocess.run([cmd], shell=True, check=True)
        except subprocess.CalledProcessError as e:
            raise (e)


def upload_pypi_packages(args, WHL_PATHS):
    """
    Takes a list of path values and uploads them to pypi using twine, using token stored in environment variable
    """
    dry_run = not args.upload

    # Note: TWINE_USERNAME and TWINE_PASSWORD are expected to be set in the environment
    options = "--username __token__ "

    if args.test_pypi:
        options += "--repository-url https://test.pypi.org/legacy/ "
        # TODO:
        # maybe "--repository testpypi " works the same (and shorter)?
        # Ref: https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives

    for dist_path in WHL_PATHS:
        cmd = "twine upload " + options + f" {dist_path}/*"
        exe_cmd(cmd, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload  pypi packages for PiPPy"
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        required=False,
        help="Actually upload packages; otherwise dry run",
    )

    parser.add_argument(
        "--test-pypi",
        action="store_true",
        help="Upload to test.pypi instead of pypi",
    )

    args = parser.parse_args()

    PACKAGES = ["pippy"]

    if args.upload:
        PiPPY_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "dist"))[0]
    else:
        PiPPY_WHEEL_PATH = os.path.join(REPO_ROOT, "dist")

    WHL_PATHS = [PiPPY_WHEEL_PATH]

    upload_pypi_packages(args, WHL_PATHS)
