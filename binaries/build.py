import argparse
import glob
import os
import sys


"""
WARNING:
Please make sure the "build" folder and the "dist" folder are cleaned before build.
You can achieve that by running:
`python setup.py clean`
"""

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)


def build_dist_whl(args):
    """
    Function to build the wheel files for PiPPy
    """

    print("## Started pippy build")
    create_wheel_cmd = "python setup.py bdist_wheel "

    os.chdir(REPO_ROOT)

    # Build wheel
    print(f"## In directory: {os.getcwd()} | Executing command: {create_wheel_cmd}")

    if not args.dry_run:
        build_exit_code = os.system(create_wheel_cmd)
        # If any one of the steps fail, exit with error
        if build_exit_code != 0:
            sys.exit(f"## PiPPy build Failed !")


def build(args):
    dist_dir = os.path.join(REPO_ROOT, "dist")

    # Detect whether old build exists
    # If any, stop
    if os.path.exists(dist_dir):
        raise RuntimeError(
            f"dist folder already exist at {dist_dir}. Please run: "
            "`python setup.py clean` "
            "to clean existing builds."
        )

    # Build dist wheel files
    build_dist_whl(args)

    pippy_wheel_path = os.path.join(dist_dir, "*.whl")
    if not args.dry_run:
        # `glob.glob` returns a list of files that matches the path having wildcards
        pippy_wheel_path = glob.glob(pippy_wheel_path)

    print(f"## PiPPy wheel location: {pippy_wheel_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build wheel package for pippy"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands that will be run without running them",
    )

    args = parser.parse_args()

    build(args)
