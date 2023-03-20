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

"""Implements ResNets[1] and Axial-ResNets [2, 3] as pixel encoders.

[1] Deep residual learning for image recognition.
    CVPR 2016.
      Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

[2] Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
    ECCV 2020.
      Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
      Liang-Chieh Chen.

[3] MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
    CVPR 2021.
      Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
"""

import functools

import tensorflow as tf

from deeplab2.model.encoder import axial_resnet

resnet50 = functools.partial(
    axial_resnet.AxialResNet,
    output_stride=32,
    classification_mode=True,
    backbone_type="resnet",
    use_axial_beyond_stride=0,
    backbone_use_transformer_beyond_stride=0,
    activation="relu")

# This is the same backbone as MaX-S, which uses Inception Stem and
# incorporates Axial-Attention in the last two stages of ResNet-50.
axial_resnet50 = functools.partial(
    axial_resnet.AxialResNet,
    output_stride=32,
    classification_mode=True,
    backbone_type="resnet_beta",
    use_axial_beyond_stride=16,
    backbone_use_transformer_beyond_stride=0,
    activation="gelu")


def get_model(model_name, input_shape, drop_path_keep_prob=1.0, **kwargs):
  """Gets an (Axial-)ResNet model."""
  block_group_config = {
      "drop_path_schedule": "linear",
      "drop_path_keep_prob": drop_path_keep_prob
  }
  model_name = model_name.lower()
  if model_name == "resnet50":
    model = resnet50(
        name=model_name, block_group_config=block_group_config, **kwargs)
  elif model_name == "axial_resnet50":
    model = axial_resnet50(
        name=model_name, block_group_config=block_group_config, **kwargs)
  else:
    raise ValueError("Unsupported backbone %s!" % model_name)

  # Build the model.
  model(tf.keras.Input(shape=input_shape))

  return model
