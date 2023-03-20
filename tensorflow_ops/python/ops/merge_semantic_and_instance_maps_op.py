# coding=utf-8
# Copyright 2023 The Deeplab2 Authors.
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

"""Utility function for the C++ TensorFlow MergeSemanticAndInstanceMaps op."""

import tensorflow as tf

# Make the TensorFlow MergeSemanticAndInstanceMaps op accessible by importing
# merge_semantic_and_instance_maps_op.py.
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_merge_semantic_and_instance_maps_op = load_library.load_op_library(resource_loader.get_path_to_datafile('../../kernels/merge_semantic_and_instance_maps_op.so'))

merge_semantic_and_instance_maps = gen_merge_semantic_and_instance_maps_op.merge_semantic_and_instance_maps

tf.no_gradient('MergeSemanticAndInstanceMaps')
