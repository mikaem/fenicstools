#!/bin/bash

if [[ "$(uname)" == "Darwin" ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9
  export CXXFLAGS="-std=c++11 -stdlib=libc++ $CXXFLAGS"
fi

export INSTANT_CACHE_DIR=${PWD}/instant

$PYTHON -m pip install -v --no-deps cppimport

$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv
