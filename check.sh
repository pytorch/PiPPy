#!/bin/bash

set -e

./format.sh --check

pyre check
flake8 pippy spmd test

