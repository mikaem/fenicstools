#!/bin/bash
set -e

if [[ "$(uname)" == "Darwin" ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9
  export CXXFLAGS="-std=c++11 -stdlib=libc++ $CXXFLAGS"
fi

export INSTANT_CACHE_DIR=${PWD}/instant
export DIJITSO_CACHE_DIR=${PWD}/instant

pushd tests

# Downgrade pybind11 cause 2.2.4 is buggy
#conda install pybind11=2.2.3

$PYTHON -b -m pytest -vs

#mpirun -np 2 py.test -v
