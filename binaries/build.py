import argparse
import glob
import os
import sys

# To help discover local modules
REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(REPO_ROOT)


def build_dist_whl(args):
    """
    Function to build the wheel files for torchserve, model-archiver and workflow-archiver
    """
    if args.nightly:
        print(
            "## Started pippy build"
        )
        create_wheel_cmd = "python setup.py "
    else:
        print("## Started pippy build")
        create_wheel_cmd = "python setup.py bdist_wheel "

    cur_dir = REPO_ROOT
        
    os.chdir(cur_dir)
    cur_wheel_cmd = create_wheel_cmd
    cur_wheel_cmd = (
        create_wheel_cmd + "--override-name " + "PiPPy" + "-nightly" + " bdist_wheel"
        if args.nightly
        else create_wheel_cmd
    )

    # Build wheel
    print(f"## In directory: {os.getcwd()} | Executing command: {cur_wheel_cmd}")

    if not args.dry_run:
        build_exit_code = os.system(cur_wheel_cmd)
        # If any one of the steps fail, exit with error
        if build_exit_code != 0:
            sys.exit(f"## PiPPy build Failed !")


def build(args):

    # Build dist wheel files
    build_dist_whl(args)

    os.chdir(REPO_ROOT)

    if not args.dry_run:
        ts_wheel_path = glob.glob(os.path.join(REPO_ROOT, "dist", "*.whl"))[0]

    else:
        ts_wheel_path = os.path.join(REPO_ROOT, "dist", "*.whl")

    print(f"## PiPPY wheel location: {ts_wheel_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build for pippy"
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        required=False,
        help="specify nightly is being built",
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="dry_run will print the commands that will be run without running them",
    )

    args = parser.parse_args()

    build(args)