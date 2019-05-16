#!/bin/bash

if [[ "$(uname)" == "Darwin" ]]; then
  export MACOSX_DEPLOYMENT_TARGET=10.9
  export CXXFLAGS="-std=c++11 -stdlib=libc++ $CXXFLAGS"
fi

export INSTANT_CACHE_DIR=${PWD}/instant

pip install --no-deps --no-binary :all: cppimport

python setup.py install
