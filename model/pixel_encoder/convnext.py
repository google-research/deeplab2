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

"""ConvNeXt [1] backbone translated from official PyTorch Implementation [2].

[1] A ConvNet for the 2020s,
    CVPR 2022.
      Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer,
      Trevor Darrell, Saining Xie.
[2] https://github.com/facebookresearch/ConvNeXt
"""

import tensorflow as tf

from deeplab2.model.layers import drop_path


def conv_2d(*args, **kwargs):
  return tf.keras.layers.Conv2D(
      kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.02),
      bias_initializer="zeros", *args, **kwargs)


def dense(*args, **kwargs):
  return tf.keras.layers.Dense(
      kernel_initializer=tf.keras.initializers.truncated_normal(stddev=0.02),
      bias_initializer="zeros", *args, **kwargs)


class ConvNeXtBlock(tf.keras.Model):
  """ConvNeXt block."""

  def __init__(self,
               dim,
               drop_path_keep_prob=0.0,
               layer_scale_init_value=1e-6,
               **kwargs):
    super().__init__(**kwargs)
    self.depthwise_conv = conv_2d(
        filters=dim, kernel_size=7, padding="same", groups=dim)
    self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.pointwise_conv1 = dense(4 * dim)
    self.act = tf.keras.layers.Activation("gelu")
    self.pointwise_conv2 = dense(dim)
    if layer_scale_init_value > 0:
      self.gamma = self.add_weight(
          name="layer_scale",
          shape=(1, 1, 1, dim),
          initializer=tf.keras.initializers.Constant(layer_scale_init_value))
    else:
      self.gamma = None
    self.drop_path = (
        drop_path.DropPath(drop_path_keep_prob)
        if drop_path_keep_prob < 1.0 else tf.keras.layers.Activation("linear"))

  def call(self, x):
    inputs = x

    x = self.depthwise_conv(x)
    x = self.norm(x)
    x = self.pointwise_conv1(x)
    x = self.act(x)
    x = self.pointwise_conv2(x)

    if self.gamma is not None:
      x = self.gamma * x

    x = inputs + self.drop_path(x)
    return x


class ConvNeXt(tf.keras.Model):
  """ConvNeXt model."""

  def __init__(self,
               name,
               depths=(3, 3, 9, 3),
               dims=(96, 192, 384, 768),
               drop_path_keep_prob=1.0,
               layer_scale_init_value=1e-6,
               zero_padding_for_downstream=False):
    """Initializes a ConvNeXt backbone.

    Args:
      name: A string, the name of the model.
      depths: A list of integers specifying number of blocks for each stage.
      dims: A list of integers specifying number of channels for each stage.
      drop_path_keep_prob: A float specifying the keep probability of drop path.
      layer_scale_init_value: A float specifying the initialization value of
        layer scale.
      zero_padding_for_downstream: A boolean speicifying whether additional zero
        padding needs to be performed before each downsampling for downstream
        tasks.
    """
    super().__init__(name=name)

    self.downsample_layers = []
    stem = [
        conv_2d(dims[0], kernel_size=4, strides=4),
        tf.keras.layers.LayerNormalization(epsilon=1e-6),
    ]
    if zero_padding_for_downstream:
      # This is needed as the kernel size and stride ConvNeXt uses for
      # downsampling cannot directly handle odd-sized inputs for downstream
      # tasks in our framework. Therefore, we pad 1 zero on the top and left
      # regions, and pad 2 zeros on the bottom and right regions.
      padding = (1, 2)
      stem = [tf.keras.layers.ZeroPadding2D(padding=(padding, padding))] + stem
    stem = tf.keras.Sequential(stem, name="stem")
    self.downsample_layers.append(stem)
    for i in range(3):
      downsample_layer = [
          tf.keras.layers.LayerNormalization(epsilon=1e-6),
          conv_2d(dims[i + 1], kernel_size=2, strides=2),
      ]
      if zero_padding_for_downstream:
        # This is needed as the kernel size and stride ConvNeXt uses for
        # downsampling cannot directly handle odd-sized inputs for downstream
        # tasks in our framework. Therefore, we pad 0 zero on the top and left
        # regions, and pad 1 zeros on the bottom and right regions.
        padding = (0, 1)
        downsample_layer = [
            tf.keras.layers.ZeroPadding2D(padding=(padding, padding))
        ] + downsample_layer
      downsample_layer = tf.keras.Sequential(
          downsample_layer, name=f"downsampling_block_{i}")
      self.downsample_layers.append(downsample_layer)

    self.stages = []
    keep_rates = list(tf.linspace(1.0, drop_path_keep_prob, sum(depths)))
    cur = 0
    for i in range(4):
      stage = []
      for j in range(depths[i]):
        block = ConvNeXtBlock(
            dim=dims[i],
            drop_path_keep_prob=keep_rates[cur + j],
            layer_scale_init_value=layer_scale_init_value,
            name=f"convnext_block_{i}_{j}")
        stage.append(block)
      stage = tf.keras.Sequential(stage, name=f"convnext_stage_{i}")
      self.stages.append(stage)
      cur += depths[i]

    self._act = tf.keras.layers.Activation("gelu")

  def call(self, x):
    endpoints = {}
    for i in range(4):
      x = self.downsample_layers[i](x)
      if i == 0:
        # Stem output.
        endpoints["stage1"] = x
        endpoints["res1"] = self._act(x)
      x = self.stages[i](x)
      endpoints["stage{}".format(i + 2)] = x
      endpoints["res{}".format(i + 2)] = self._act(x)
    return endpoints


def get_model(model_name, input_shape, pretrained_weights_path=None, **kwargs):
  """Gets a ConvNeXt model."""
  model_name = model_name.lower()
  if model_name == "convnext_tiny":
    model = ConvNeXt(name=model_name,
                     depths=[3, 3, 9, 3],
                     dims=[96, 192, 384, 768],
                     **kwargs)
  elif model_name == "convnext_small":
    model = ConvNeXt(name=model_name,
                     depths=[3, 3, 27, 3],
                     dims=[96, 192, 384, 768],
                     **kwargs)
  elif model_name == "convnext_base":
    model = ConvNeXt(name=model_name,
                     depths=[3, 3, 27, 3],
                     dims=[128, 256, 512, 1024],
                     **kwargs)
  elif model_name == "convnext_large":
    model = ConvNeXt(name=model_name,
                     depths=[3, 3, 27, 3],
                     dims=[192, 384, 768, 1536],
                     **kwargs)
  elif model_name == "convnext_xlarge":
    model = ConvNeXt(name=model_name,
                     depths=[3, 3, 27, 3],
                     dims=[256, 512, 1024, 2048],
                     **kwargs)
  else:
    raise ValueError("Unsupported model name for ConvNeXt!")

  # Build the model.
  model(tf.keras.Input(shape=input_shape))

  if pretrained_weights_path:
    model.load_weights(pretrained_weights_path)

  return model
