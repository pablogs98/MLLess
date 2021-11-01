#======================================================================================================================
# Script name:  compile_linux_cython.sh
# Usage:        ./compile_linux_cython.sh
# Description:  Cython module compiling script for x86-64 Linux (IBM Cloud functions' architecture)
#======================================================================================================================

if CONTAINER_ID=$(docker run --pull missing -t -d python:3.8 /bin/bash); then
  if docker exec "$CONTAINER_ID" git clone https://github.com/pablogs98/MLLess; then
    docker exec -w /MLLess "$CONTAINER_ID" pip3 install -r ./lithops_runtime/linux_compile/requirements.txt
    docker exec -w /MLLess "$CONTAINER_ID" python3 setup.py build_ext --inplace
    docker cp "$CONTAINER_ID":/MLLess/lithops_runtime/model.cpython-38-x86_64-linux-gnu.so .
    docker cp "$CONTAINER_ID":/MLLess/lithops_runtime/mf_model.cpython-38-x86_64-linux-gnu.so .
    docker cp "$CONTAINER_ID":/MLLess/lithops_runtime/lr_model.cpython-38-x86_64-linux-gnu.so .
    docker cp "$CONTAINER_ID":/MLLess/lithops_runtime/sparse_lr_model.cpython-38-x86_64-linux-gnu.so .
  else
    return 1
  fi
  docker container rm -f "$CONTAINER_ID"
  return 0
else
  echo "Docker is not running! Run docker and try again."
  return 1
fi
