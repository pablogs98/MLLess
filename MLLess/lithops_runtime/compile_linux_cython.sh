#======================================================================================================================
# Script name:  compile_linux_cython.sh
# Usage:        ./compile_linux_cython.sh
# Description:  Cython module compiling script for x86-64 Linux (IBM Cloud functions' architecture)
#======================================================================================================================

if CONTAINER_ID=$(docker run --pull missing -t -d python:3.8 /bin/bash); then
  if docker exec -it "$CONTAINER_ID" git clone https://github.com/pablogs98/MLLess; then
    docker exec -it -w /MLLess "$CONTAINER_ID" pip3 install -r ./lithops_runtime/linux_compile/requirements.txt
    docker exec -it -w /MLLess "$CONTAINER_ID" python3 setup.py build_ext --inplace
    docker cp "$CONTAINER_ID":/MLLess/lithops_runtime/model.cpython-38-x86_64-linux-gnu.so .
    docker cp "$CONTAINER_ID":/MLLess/lithops_runtime/mf_model.cpython-38-x86_64-linux-gnu.so .
    docker cp "$CONTAINER_ID":/MLLess/lithops_runtime/lr_model.cpython-38-x86_64-linux-gnu.so .
    docker cp "$CONTAINER_ID":/MLLess/lithops_runtime/sparse_lr_model.cpython-38-x86_64-linux-gnu.so .
  else
    echo "There was an error cloning the repository."
    exit 1
  fi
  echo "Successfully compiled and copied MLLess' Cython binaries."
  docker container rm -f "$CONTAINER_ID"
  exit 0
else
  echo "Docker is not running! Run docker and try again."
  exit 1
fi
