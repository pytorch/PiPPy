#!/bin/bash

set -e

source .venv/bin/activate

./format.sh --check

pyre check

flake8 pippy spmd test/spmd

mypy spmd test/spmd

