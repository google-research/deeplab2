# coding=utf-8
# Copyright 2021 The Deeplab2 Authors.
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

"""This file contains functions to build encoder and decoder."""
import tensorflow as tf

from deeplab2 import config_pb2
from deeplab2.model.decoder import deeplabv3
from deeplab2.model.decoder import deeplabv3plus
from deeplab2.model.decoder import max_deeplab
from deeplab2.model.decoder import motion_deeplab_decoder
from deeplab2.model.decoder import panoptic_deeplab
from deeplab2.model.decoder import vip_deeplab_decoder
from deeplab2.model.encoder import axial_resnet_instances
from deeplab2.model.encoder import mobilenet


def create_encoder(backbone_options: config_pb2.ModelOptions.BackboneOptions,
                   bn_layer: tf.keras.layers.Layer,
                   conv_kernel_weight_decay: float = 0.0) -> tf.keras.Model:
  """Creates an encoder.

  Args:
    backbone_options: A proto config of type
      config_pb2.ModelOptions.BackboneOptions.
    bn_layer: A tf.keras.layers.Layer that computes the normalization.
    conv_kernel_weight_decay: A float, the weight decay for convolution kernels.

  Returns:
    An instance of tf.keras.Model containing the encoder.

  Raises:
    ValueError: An error occurs when the specified encoder meta architecture is
      not supported.
  """
  if ('resnet' in backbone_options.name or
      'swidernet' in backbone_options.name or
      'axial_deeplab' in backbone_options.name or
      'max_deeplab' in backbone_options.name):
    return create_resnet_encoder(
        backbone_options,
        bn_layer=bn_layer,
        conv_kernel_weight_decay=conv_kernel_weight_decay)
  elif 'mobilenet' in backbone_options.name:
    return create_mobilenet_encoder(
        backbone_options,
        bn_layer=bn_layer,
        conv_kernel_weight_decay=conv_kernel_weight_decay)
  raise ValueError('The specified encoder %s is not a valid encoder.' %
                   backbone_options.name)


def create_mobilenet_encoder(
    backbone_options: config_pb2.ModelOptions.BackboneOptions,
    bn_layer: tf.keras.layers.Layer,
    conv_kernel_weight_decay: float = 0.0) -> tf.keras.Model:
  """Creates a MobileNet encoder specified by name.

  Args:
    backbone_options: A proto config of type
      config_pb2.ModelOptions.BackboneOptions.
    bn_layer: A tf.keras.layers.Layer that computes the normalization.
    conv_kernel_weight_decay: A float, the weight decay for convolution kernels.

  Returns:
    An instance of tf.keras.Model containing the MobileNet encoder.
  """
  if backbone_options.name.lower() == 'mobilenet_v3_large':
    backbone = mobilenet.MobileNetV3Large
  elif backbone_options.name.lower() == 'mobilenet_v3_small':
    backbone = mobilenet.MobileNetV3Small
  else:
    raise ValueError('The specified encoder %s is not a valid encoder.' %
                     backbone_options.name)
  assert backbone_options.use_squeeze_and_excite
  assert backbone_options.drop_path_keep_prob == 1
  assert backbone_options.use_sac_beyond_stride == -1
  assert backbone_options.backbone_layer_multiplier == 1
  return backbone(
      output_stride=backbone_options.output_stride,
      width_multiplier=backbone_options.backbone_width_multiplier,
      bn_layer=bn_layer,
      conv_kernel_weight_decay=conv_kernel_weight_decay)


def create_resnet_encoder(
    backbone_options: config_pb2.ModelOptions.BackboneOptions,
    bn_layer: tf.keras.layers.Layer,
    conv_kernel_weight_decay: float = 0.0) -> tf.keras.Model:
  """Creates a ResNet encoder specified by name.

  Args:
    backbone_options: A proto config of type
      config_pb2.ModelOptions.BackboneOptions.
    bn_layer: A tf.keras.layers.Layer that computes the normalization.
    conv_kernel_weight_decay: A float, the weight decay for convolution kernels.

  Returns:
    An instance of tf.keras.Model containing the ResNet encoder.
  """
  return axial_resnet_instances.get_model(
      backbone_options.name,
      output_stride=backbone_options.output_stride,
      stem_width_multiplier=backbone_options.stem_width_multiplier,
      width_multiplier=backbone_options.backbone_width_multiplier,
      backbone_layer_multiplier=backbone_options.backbone_layer_multiplier,
      block_group_config={
          'use_squeeze_and_excite': backbone_options.use_squeeze_and_excite,
          'drop_path_keep_prob': backbone_options.drop_path_keep_prob,
          'drop_path_schedule': backbone_options.drop_path_schedule,
          'use_sac_beyond_stride': backbone_options.use_sac_beyond_stride},
      bn_layer=bn_layer,
      conv_kernel_weight_decay=conv_kernel_weight_decay)


def create_decoder(model_options: config_pb2.ModelOptions,
                   bn_layer: tf.keras.layers.Layer,
                   ignore_label: int) -> tf.keras.Model:
  """Creates a DeepLab decoder.

  Args:
    model_options: A proto config of type config_pb2.ModelOptions.
    bn_layer: A tf.keras.layers.Layer that computes the normalization.
    ignore_label: An integer specifying the ignore label.

  Returns:
    An instance of tf.keras.layers.Layer containing the decoder.

  Raises:
    ValueError: An error occurs when the specified meta architecture is not
      supported.
  """
  meta_architecture = model_options.WhichOneof('meta_architecture')
  if meta_architecture == 'deeplab_v3':
    return deeplabv3.DeepLabV3(
        model_options.decoder, model_options.deeplab_v3, bn_layer=bn_layer)
  elif meta_architecture == 'deeplab_v3_plus':
    return deeplabv3plus.DeepLabV3Plus(
        model_options.decoder, model_options.deeplab_v3_plus, bn_layer=bn_layer)
  elif meta_architecture == 'panoptic_deeplab':
    return panoptic_deeplab.PanopticDeepLab(
        model_options.decoder,
        model_options.panoptic_deeplab,
        bn_layer=bn_layer)
  elif meta_architecture == 'motion_deeplab':
    return motion_deeplab_decoder.MotionDeepLabDecoder(
        model_options.decoder,
        model_options.motion_deeplab,
        bn_layer=bn_layer)
  elif meta_architecture == 'vip_deeplab':
    return vip_deeplab_decoder.ViPDeepLabDecoder(
        model_options.decoder,
        model_options.vip_deeplab,
        bn_layer=bn_layer)
  elif meta_architecture == 'max_deeplab':
    return max_deeplab.MaXDeepLab(
        model_options.decoder,
        model_options.max_deeplab,
        ignore_label=ignore_label,
        bn_layer=bn_layer)
  raise ValueError('The specified meta architecture %s is not implemented.' %
                   meta_architecture)
