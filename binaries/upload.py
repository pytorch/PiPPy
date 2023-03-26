#! /usr/env/bin
import argparse
import glob
import os
import sys

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)

def try_and_handle(cmd, dry_run=False):
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
    dry_run = args.dry_run

    # Note: TWINE_USERNAME and TWINE_PASSWORD are expected to be set in the environment
    for dist_path in WHL_PATHS:
        if args.test_pypi:
            try_and_handle(
                f"twine upload {dist_path}/* --username __token__ --repository-url https://test.pypi.org/legacy/",
                dry_run,
            )
        else:
            try_and_handle(f"twine upload --username __token__ {dist_path}/*", dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload anaconda and pypi packages for torchserve and torch-model-archiver"
    )
    parser.add_argument(
        "--upload-conda-packages",
        action="store_true",
        required=False,
        help="Specify whether to upload conda packages",
    )
    parser.add_argument(
        "--upload-pypi-packages",
        action="store_true",
        required=False,
        help="Specify whether to upload pypi packages",
    )
    parser.add_argument(
        "--test-pypi",
        action="store_true",
        required=False,
        help="Specify whether to upload to test PyPI",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="dry_run will print the commands that will be run without running them. Only works for pypi now",
    )
    args = parser.parse_args()

    PACKAGES = ["torchserve"]
    

    if not args.dry_run:
        PiPPY_WHEEL_PATH = glob.glob(os.path.join(REPO_ROOT, "dist"))[0]
    else:
        PiPPY_WHEEL_PATH = os.path.join(REPO_ROOT, "dist")
      

    WHL_PATHS = [PiPPY_WHEEL_PATH]

    if args.upload_pypi_packages:
        upload_pypi_packages(args, WHL_PATHS)
