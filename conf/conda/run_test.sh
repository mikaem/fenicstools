#!/bin/bash
set -e

pushd "$SRC_DIR/tests"
python -b -m pytest -vs

#mpirun -np 2 py.test -v
