#!/bin/bash

set -e

./format.sh --check

pyre check

flake8 pippy spmd test/spmd

mypy spmd test/spmd

pylint --disable=all --enable=unused-import $(git ls-files '*.py')

