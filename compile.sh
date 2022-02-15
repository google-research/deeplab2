# Copyright 2022 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Quick start command line to setup deeplab2 (Linux only).
# Example command to run:
#   deeplab2/compile.sh ${PATH_TO_PROTOC}
#
# This script assumes that the following folder structure:
#
#   + root
#    + deeplab2
#    + models
#      + orbit
#    + cocoapi
#      + PythonAPI
#
# Besides, the script also assumes that `protoc` can be accessed from command
# line.

#!/bin/bash

set -e

# cpu or gpu
CONFIG="cpu"

function tolower() {
  echo "${1,,}"
}

if [[ ! -z "$1" ]]
then
  echo "Setting configuration from argument($1)..."
  CONFIG=$(tolower "$1")
  if  [ "$CONFIG" != "cpu" ] && [ "$CONFIG" != "gpu" ]
  then
    echo "Configuration must be either \"cpu\" or \"gpu\", exiting..."
    exit 1
  fi
fi

echo "Running configuration with $CONFIG."

# Protobuf compilation
# Replace `protoc` with `${PATH_TO_PROTOC}` if protobuf compilier is downloaded
# from web.
echo "-----------------------------------------------------------------------"
echo "Compiling protobuf..."
echo "-----------------------------------------------------------------------"
protoc deeplab2/*.proto --python_out=.

# Compile custom ops
# See details in https://www.tensorflow.org/guide/create_op#compile_the_op_using_your_system_compiler_tensorflow_binary_installation
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
OP_NAME='deeplab2/tensorflow_ops/kernels/merge_semantic_and_instance_maps_op'

if [ "$CONFIG" == "cpu" ]
then
  # CPU
  echo "-----------------------------------------------------------------------"
  echo "Compiling the custom cc op: merge_semantic_and_instance_maps_op (CPU)..."
  echo "-----------------------------------------------------------------------"
  g++ -std=c++14 -shared \
  ${OP_NAME}.cc ${OP_NAME}_kernel.cc -o ${OP_NAME}.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2
else
  # GPU
  # (https://www.tensorflow.org/guide/create_op#compiling_the_kernel_for_the_gpu_device)
  echo "-----------------------------------------------------------------------"
  echo "Compiling the custom cc op: merge_semantic_and_instance_maps_op (GPU)..."
  echo "-----------------------------------------------------------------------"
  nvcc -std=c++14 -c -o ${OP_NAME}_kernel.cu.o \
  ${OP_NAME}_kernel.cu.cc \
    ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr

  g++ -std=c++14 -shared -o ${OP_NAME}.so ${OP_NAME}.cc ${OP_NAME}_kernel.cc \
    ${OP_NAME}_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
fi

# PYTHONPATH
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/models:`pwd`/cocoapi/PythonAPI

# Runing test
echo "-----------------------------------------------------------------------"
echo "Running tests for merge_semantic_and_instance_maps_op..."
echo "-----------------------------------------------------------------------"
python deeplab2/tensorflow_ops/python/kernel_tests/merge_semantic_and_instance_maps_op_test.py

# End-to-end tests
echo "-----------------------------------------------------------------------"
echo "Running end-to-end tests..."
echo "-----------------------------------------------------------------------"

# Model training test (test for custom ops, protobug)
python deeplab2/model/deeplab_test.py

# Model evaluation test (test for other packages such as orbit, cocoapi, etc)
python deeplab2/trainer/evaluator_test.py

echo "------------------------"
echo "Done with configuration!"
echo "------------------------"

