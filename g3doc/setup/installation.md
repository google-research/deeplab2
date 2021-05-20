# DeepLab2

## **Requirements**

DeepLab2 depends on the following libraries:

*   Python3
*   Numpy
*   Pillow
*   Matplotlib
*   Tensorflow 2.5
*   Cython
*   [Google Protobuf](https://developers.google.com/protocol-buffers)
*   [Orbit](https://github.com/tensorflow/models/tree/master/orbit)
*   [pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
    (for AP-Mask)

## **Installation**

### Git Clone the Project

Clone the
[`google-research/deeplab2`](https://github.com/google-research/deeplab2)
repository.

```bash
mkdir ${YOUR_PROJECT_NAME}
cd ${YOUR_PROJECT_NAME}
git clone https://github.com/google-research/deeplab2.git
```

### **Option A: Direct Installation**

### Install TensorFlow 2.5 via PIP

```bash
# Should come with compatible numpy package.
pip install tensorflow==2.5
```

### Install Protobuf

Below is a quick-to-start command line to install
[protobuf](https://github.com/protocolbuffers/protobuf) in Linux:

```bash
sudo apt-get install protobuf-compiler
```

Alternatively, you can also download the package from web on other platforms.
Please refer to https://github.com/protocolbuffers/protobuf for more details
about installation.

### Other Libraries

The remaining libraries can be installed via pip:

```bash
# Pillow
pip install pillow
# matplotlib
pip install matplotlib
# Cython
pip install cython
```

### Install Orbit

[`Orbit`](https://github.com/tensorflow/models/tree/master/orbit) is a flexible,
lightweight library designed to make it easy to write custom training loops in
TensorFlow 2. We used Orbit in our train/eval loops. You need to download the
code below:

```bash
cd ${YOUR_PROJECT_NAME}
git clone https://github.com/tensorflow/models.git
```

### Install pycocotools

We also use
[`pycocotools`](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
for instance segmentation evaluation. Below is the installation guide:

```bash
cd ${YOUR_PROJECT_NAME}
git clone https://github.com/cocodataset/cocoapi.git

# Compile cocoapi
cd ${YOUR_PROJECT_NAME}/cocoapi/PythonAPI
make
cd ${YOUR_PROJECT_NAME}
```

## **Compilation**

The following instructions are running from `${YOUR_PROJECT_NAME}` directory:

```bash
cd ${YOUR_PROJECT_NAME}
```

### Add Libraries to PYTHONPATH

When running locally, `${YOUR_PROJECT_NAME}` directory should be appended to
PYTHONPATH. This can be done by running the following command:

```bash
# From ${YOUR_PROJECT_NAME}:

# deeplab2
export PYTHONPATH=$PYTHONPATH:`pwd`
# orbit
export PYTHONPATH=$PYTHONPATH:${PATH_TO_MODELS}
# pycocotools
export PYTHONPATH=$PYTHONPATH:${PATH_TO_cocoapi_PythonAPI}
```

If you clone `models(/orbit)` and `cocoapi` under `${YOUR_PROJECT_NAME}`, here
is an example:

```bash
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/models:`pwd`/cocoapi/PythonAPI
```

### Compile Protocol Buffers

In DeepLab2, we define
[protocol buffers](https://developers.google.com/protocol-buffers) to configure
training and evaluation variants (`deeplab2/config.proto`). However, protobuf
needs to be compiled beforehand into a python recognizable format. To compile
protobuf, run:

```bash
# `${PATH_TO_PROTOC}` is the directory where the `protoc` binary locates.
${PATH_TO_PROTOC} deeplab2/*.proto --python_out=.

# Alternatively, if protobuf compiler is globally accessible, you can simply run:
protoc deeplab2/*.proto --python_out=.
```

### (Optional) Compile Custom Ops

We implemented efficient merging operation to merge semantic and instance maps
for fast inference. You can follow the guide below to compile the provided
efficient merging operation in c++ under the folder `tensorflow_ops`.

The script is mostly from
[Compile the op using your system compiler](https://www.tensorflow.org/guide/create_op#compile_the_op_using_your_system_compiler_tensorflow_binary_installation)
in the official tensorflow guide to create custom ops. Please refer to
[Create an op](https://www.tensorflow.org/guide/create_op#compile_the_op_using_your_system_compiler_tensorflow_binary_installation)
for more details.

```bash
TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )
OP_NAME='deeplab2/tensorflow_ops/kernels/merge_semantic_and_instance_maps_op'

# CPU
g++ -std=c++11 -shared \
${OP_NAME}.cc ${OP_NAME}_kernel.cc -o ${OP_NAME}.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -O2

# GPU support (https://www.tensorflow.org/guide/create_op#compiling_the_kernel_for_the_gpu_device)
nvcc -std=c++11 -c -o ${OP_NAME}_kernel.cu.o ${OP_NAME}_kernel.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 -shared -o ${OP_NAME}.so ${OP_NAME}.cc ${OP_NAME}_kernel.cc \
  ${OP_NAME}_kernel.cu.o ${TF_CFLAGS[@]} -fPIC -lcudart ${TF_LFLAGS[@]}
```

To test if the compilation is done successfully, you can run:

```bash
python deeplab2/tensorflow_ops/python/kernel_tests/merge_semantic_and_instance_maps_op_test.py
```

Optionally, you could set `merge_semantic_and_instance_with_tf_op` to `false` in
the config file to skip provided efficient merging operation and use the slower
pure TF functions instead. See
`deeplab2/configs/cityscaspes/panoptic_deeplab/resnet50_os32_merge_with_pure_tf_func.textproto`
as an example.

### Test the Configuration

You can test if you have successfully installed and configured DeepLab2 by
running the following commands (requires compilation of custom ops):

```bash
# Model training test (test for custom ops, protobuf)
python deeplab2/model/deeplab_test.py

# Model evaluator test (test for other packages such as orbit, cocoapi, etc)
python deeplab2/trainer/evaluator_test.py
```

### Quick All-in-One Script for Compilation (Linux Only)

We also provide a shell script to help you quickly compile and test everything
mentioned above for Linux users:

```bash
# CPU
deeplab2/compile.sh

# GPU
deeplab2/compile.sh gpu
```


## Troubleshooting

**Q1: Can I use [conda](https://anaconda.org/) instead of pip?**

**A1:** We experienced several dependency issues with the most recent conda package. We therefore do not provide support for installing deeplab2 via conda at this stage.