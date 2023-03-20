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

"""MobileNetV3 models for Deep Labeling.

Reference:
  Howard, A., Sandler, M., et al. Searching for mobilenetv3. In ICCV, 2019
"""
from typing import Any, Callable, Mapping, Optional, Sequence

import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import blocks
from deeplab2.model.layers import convolutions

# The default input image channels.
_INPUT_CHANNELS = 3


MNV3Small_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3Small',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'is_endpoint'],
    'block_specs': [
        ('conv_bn', 3, 2, 16,
         'hard_swish', None, None, True),
        ('inverted_bottleneck', 3, 2, 16,
         'relu', 0.25, 1, True),
        ('inverted_bottleneck', 3, 2, 24,
         'relu', None, 72. / 16, False),
        ('inverted_bottleneck', 3, 1, 24,
         'relu', None, 88. / 24, True),
        ('inverted_bottleneck', 5, 2, 40,
         'hard_swish', 0.25, 4., False),
        ('inverted_bottleneck', 5, 1, 40,
         'hard_swish', 0.25, 6., False),
        ('inverted_bottleneck', 5, 1, 40,
         'hard_swish', 0.25, 6., False),
        ('inverted_bottleneck', 5, 1, 48,
         'hard_swish', 0.25, 3., False),
        ('inverted_bottleneck', 5, 1, 48,
         'hard_swish', 0.25, 3., True),
        ('inverted_bottleneck', 5, 2, 96,
         'hard_swish', 0.25, 6., False),
        ('inverted_bottleneck', 5, 1, 96,
         'hard_swish', 0.25, 6., False),
        ('inverted_bottleneck', 5, 1, 96,
         'hard_swish', 0.25, 6., False),
        ('conv_bn', 1, 1, 576,
         'hard_swish', None, None, True),
    ]
}


MNV3Large_BLOCK_SPECS = {
    'spec_name': 'MobileNetV3Large',
    'block_spec_schema': ['block_fn', 'kernel_size', 'strides', 'filters',
                          'activation', 'se_ratio', 'expand_ratio',
                          'is_endpoint'],
    'block_specs': [
        ('conv_bn', 3, 2, 16,
         'hard_swish', None, None, False),
        ('inverted_bottleneck', 3, 1, 16,
         'relu', None, 1., True),
        ('inverted_bottleneck', 3, 2, 24,
         'relu', None, 4., False),
        ('inverted_bottleneck', 3, 1, 24,
         'relu', None, 3., True),
        ('inverted_bottleneck', 5, 2, 40,
         'relu', 0.25, 3., False),
        ('inverted_bottleneck', 5, 1, 40,
         'relu', 0.25, 3., False),
        ('inverted_bottleneck', 5, 1, 40,
         'relu', 0.25, 3., True),
        ('inverted_bottleneck', 3, 2, 80,
         'hard_swish', None, 6., False),
        ('inverted_bottleneck', 3, 1, 80,
         'hard_swish', None, 2.5, False),
        ('inverted_bottleneck', 3, 1, 80,
         'hard_swish', None, 2.3, False),
        ('inverted_bottleneck', 3, 1, 80,
         'hard_swish', None, 2.3, False),
        ('inverted_bottleneck', 3, 1, 112,
         'hard_swish', 0.25, 6., False),
        ('inverted_bottleneck', 3, 1, 112,
         'hard_swish', 0.25, 6., True),
        ('inverted_bottleneck', 5, 2, 160,
         'hard_swish', 0.25, 6., False),
        ('inverted_bottleneck', 5, 1, 160,
         'hard_swish', 0.25, 6., False),
        ('inverted_bottleneck', 5, 1, 160,
         'hard_swish', 0.25, 6., False),
        ('conv_bn', 1, 1, 960,
         'hard_swish', None, None, True),
    ]
}


SUPPORTED_SPECS_MAP = {
    'MobileNetV3Large': MNV3Large_BLOCK_SPECS,
    'MobileNetV3Small': MNV3Small_BLOCK_SPECS,
}


# pylint: disable=invalid-name
def _block_spec_decoder(
    specs: Mapping[Any, Any],
    width_multiplier: float,
    divisible_by: int = 8,
    reduce_last_block_filters: bool = False) -> Sequence[Mapping[str, Any]]:
  """Decodes specs for a block.

  Args:
    specs: A `dict` specification of block specs of a mobilenet version.
    width_multiplier: A `float` multiplier for the filter size for all
      convolution ops. The value must be greater than zero. Typical usage will
      be to set this value in (0, 1) to reduce the number of parameters or
      computation cost of the model.
    divisible_by: An `int` that ensures all inner dimensions are divisible by
      this number.
    reduce_last_block_filters: A bool indicates whether to reduce the final
      block's filters.

  Returns:
    A list of block spec in dictionary that defines structure of the layers.
  """

  spec_name = specs['spec_name']
  block_spec_schema = specs['block_spec_schema']
  block_specs = specs['block_specs']

  if not block_specs:
    raise ValueError(
        'The block spec cannot be empty for {} !'.format(spec_name))

  if len(block_specs[0]) != len(block_spec_schema):
    raise ValueError('The block spec values {} do not match with '
                     'the schema {}'.format(block_specs[0], block_spec_schema))

  decoded_specs = []

  for spec in block_specs:
    spec_dict = dict(zip(block_spec_schema, spec))
    decoded_specs.append(spec_dict)

  block_id = 1
  first_layer = False
  for ds in decoded_specs:
    ds['filters'] = utils.make_divisible(
        value=ds['filters'] * width_multiplier,
        divisor=divisible_by,
        min_value=8)
    if block_id >= 5 and reduce_last_block_filters:
      ds['filters'] = ds['filters'] // 2
      if first_layer:
        ds['expand_ratio'] = ds['expand_ratio'] // 2
    block_id = block_id + 1 if ds['is_endpoint'] else block_id
    first_layer = ds['is_endpoint']

  return decoded_specs
# pylint: enable=invalid-name


class MobileNet(tf.keras.Model):
  """Creates a MobileNetV3 family model."""

  def __init__(
      self,
      model_id: str = 'MobileNetV3Small',
      width_multiplier: float = 1.0,
      output_stride: Optional[int] = None,
      min_width: int = 8,
      divisible_by: int = 8,
      regularize_depthwise: bool = False,
      bn_layer: Callable[..., Any] = tf.keras.layers.BatchNormalization,
      conv_kernel_weight_decay: float = 0.0,
      reduce_last_block_filters: bool = False,
      name: str = 'MobilenNetV3'):
    """Initializes a MobileNet V3 model.

    Args:
      model_id: A `str` of MobileNet version. The supported values are
        `MobileNetV3Large`, `MobileNetV3Small`.
      width_multiplier: A `float` of multiplier for the filters (number of
        channels) for all convolution ops. The value must be greater than zero.
        Typical usage will be to set this value in (0, 1) to reduce the number
        of parameters or computation cost of the model.
      output_stride: An `int` that specifies the requested ratio of input to
        output spatial resolution. If not None, then we invoke atrous
        convolution if necessary to prevent the network from reducing the
        spatial resolution of activation maps. The output_stride should be
        divisible by 4.
      min_width: An `int` of minimum width (number of channels) for all
        convolution ops. Enforced when width_multiplier < 1, and not an active
        constraint when width_multiplier >= 1.
      divisible_by: An `int` that ensures all intermediate feature dimensions
        are divisible by this number.
      regularize_depthwise: If True, apply regularization on depthwise conv.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.
      reduce_last_block_filters: A bool indicates whether to reduce the final
        block's filters by a factor of 2.
      name: Model name.

    Raises:
      ValueError: The MobileNet version is not supported.
      ValueError: width_multiplier is not greater than zero.
      ValueError: Output stride must be None or a multiple of 4.
      ValueError: Unknown block type i for layer j.
    """
    if model_id not in SUPPORTED_SPECS_MAP:
      raise ValueError('The MobileNet version {} '
                       'is not supported'.format(model_id))

    if width_multiplier <= 0:
      raise ValueError('width_multiplier is not greater than zero.')

    if (output_stride is not None and
        (output_stride <= 1 or (output_stride > 1 and output_stride % 4))):
      raise ValueError('Output stride must be None or a multiple of 4.')

    super().__init__(name=name)

    self._model_id = model_id
    self._width_multiplier = width_multiplier
    self._min_width = min_width
    self._output_stride = output_stride
    self._divisible_by = divisible_by
    self._regularize_depthwise = regularize_depthwise
    self._bn_layer = bn_layer
    self._conv_kernel_weight_decay = conv_kernel_weight_decay
    self._reduce_last_block_filters = reduce_last_block_filters
    self._blocks = []
    self._endpoint_names = []

    block_specs = SUPPORTED_SPECS_MAP.get(model_id)
    self._decoded_specs = _block_spec_decoder(
        specs=block_specs,
        width_multiplier=self._width_multiplier,
        divisible_by=self._divisible_by,
        reduce_last_block_filters=self._reduce_last_block_filters)

    self._mobilenet_base()

  def _mobilenet_base(self):
    """Builds the base MobileNet architecture."""

    # The current_stride variable keeps track of the output stride of the
    # activations, i.e., the running product of convolution strides up to the
    # current network layer. This allows us to invoke atrous convolution
    # whenever applying the next convolution would result in the activations
    # having output stride larger than the target output_stride.
    current_stride = 1

    # The atrous convolution rate parameter.
    rate = 1

    endpoint_level = 1
    in_filters = _INPUT_CHANNELS
    for i, block_def in enumerate(self._decoded_specs):
      # We only need to build up to 'res5' endpoint for segmentation task.
      if endpoint_level > 5:
        break

      block_name = '{}_{}'.format(block_def['block_fn'], i + 1)

      if (self._output_stride is not None and
          current_stride == self._output_stride):
        # If we have reached the target output_stride, then we need to employ
        # atrous convolution with stride=1 and multiply the atrous rate by the
        # current unit's stride for use in subsequent layers.
        layer_stride = 1
        layer_rate = rate
        rate = (
            rate * block_def['strides']
            if block_def['strides'] is not None else rate)
      else:
        layer_stride = block_def['strides']
        layer_rate = 1
        current_stride = (
            current_stride * block_def['strides']
            if block_def['strides'] is not None else current_stride)

      if block_def['block_fn'] == 'conv_bn':

        self._blocks.append(
            convolutions.Conv2DSame(
                output_channels=block_def['filters'],
                kernel_size=block_def['kernel_size'],
                strides=layer_stride,
                atrous_rate=layer_rate,
                activation=block_def['activation'],
                use_bias=False,
                bn_layer=self._bn_layer,
                use_bn=True,
                conv_kernel_weight_decay=self._conv_kernel_weight_decay,
                name=block_name,
                ))

      elif block_def['block_fn'] == 'inverted_bottleneck':
        atrous_rate = 1
        # There is no need to apply atrous convolution to any 1x1 convolution.
        if layer_rate > 1 and block_def['kernel_size'] != 1:
          atrous_rate = layer_rate
        self._blocks.append(
            blocks.InvertedBottleneckBlock(
                in_filters=in_filters,
                out_filters=block_def['filters'],
                expand_ratio=block_def['expand_ratio'],
                strides=layer_stride,
                kernel_size=block_def['kernel_size'],
                se_ratio=block_def['se_ratio'],
                activation=block_def['activation'],
                expand_se_in_filters=True,
                depthwise_activation=None,
                atrous_rate=atrous_rate,
                divisible_by=self._divisible_by,
                regularize_depthwise=self._regularize_depthwise,
                use_depthwise=True,
                # Note that whether the residual connection would be used is
                # also conditional on the in_filters and out_filters size, even
                # if use_residual=True,e.g. when input_filters != out_filters,
                # no residual connection will be created.
                use_residual=(block_def['strides'] == 1),
                bn_layer=self._bn_layer,
                conv_kernel_weight_decay=self._conv_kernel_weight_decay,
                name=block_name,
                ))

      else:
        raise ValueError('Unknown block type {} for layer {}'.format(
            block_def['block_fn'], i))

      # Register input_filters for the next level
      in_filters = block_def['filters']

      if block_def['is_endpoint']:
        # Name the endpoint to be 'res{1...5}' to align with ResNet. This
        # simplifies segmentation head implementation.
        self._endpoint_names.append('res' + str(endpoint_level))
        endpoint_level += 1
      else:
        self._endpoint_names.append(None)

  def call(self, input_tensor: tf.Tensor, training: bool = False):
    """Performs a forward pass through MobileNet."""
    net = input_tensor
    endpoints = {}
    for block, endpoint_name in zip(self._blocks, self._endpoint_names):
      if isinstance(block, blocks.InvertedBottleneckBlock):
        net, depthwise_output = block(net, training=training)
        if endpoint_name is not None:
          # Use the corresponding layer's 'depthwise_output' as the endpoint for
          # segmentation task if possible.
          endpoints[endpoint_name] = depthwise_output
      else:
        net = block(net, training=training)
        if endpoint_name is not None:
          endpoints[endpoint_name] = net
    return endpoints


def MobileNetV3Small(
    width_multiplier: float = 1.0,
    output_stride: int = 32,
    bn_layer: Callable[..., Any] = tf.keras.layers.BatchNormalization,
    conv_kernel_weight_decay: float = 0.0,
    reduce_last_block_filters: bool = False,
    name: str = 'MobileNetV3Small') -> tf.keras.Model:
  """Creates a MobileNetV3Small model.

  Args:
    width_multiplier: A float, depth_multiplier for the whole model.
    output_stride: An optional integer specifying the output stride of the
      network.
    bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
    conv_kernel_weight_decay: A float, the weight decay for convolution kernels.
    reduce_last_block_filters: A bool indicates whether to reduce the final
      block's filters by a factor of 2.
    name: Model name.

  Returns:
    The MobileNetV3Small model as an instance of tf.keras.Model.
  """
  model = MobileNet(model_id='MobileNetV3Small',
                    width_multiplier=width_multiplier,
                    output_stride=output_stride,
                    bn_layer=bn_layer,
                    conv_kernel_weight_decay=conv_kernel_weight_decay,
                    reduce_last_block_filters=reduce_last_block_filters,
                    name=name)
  return model


def MobileNetV3Large(
    width_multiplier: float = 1.0,
    output_stride: int = 32,
    bn_layer: Callable[..., Any] = tf.keras.layers.BatchNormalization,
    conv_kernel_weight_decay: float = 0.0,
    reduce_last_block_filters: bool = False,
    name: str = 'MobileNetV3Large') -> tf.keras.Model:
  """Creates a MobileNetV3Large model.

  Args:
    width_multiplier: A float, depth_multiplier for the STEM.
    output_stride: An optional integer specifying the output stride of the
      network.
    bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
    conv_kernel_weight_decay: A float, the weight decay for convolution kernels.
    reduce_last_block_filters: A bool indicates whether to reduce the final
      block's filters by a factor of 2.
    name: Model name.

  Returns:
    The MobileNetV3Large model as an instance of tf.keras.Model.
  """
  model = MobileNet(model_id='MobileNetV3Large',
                    width_multiplier=width_multiplier,
                    output_stride=output_stride,
                    bn_layer=bn_layer,
                    conv_kernel_weight_decay=conv_kernel_weight_decay,
                    reduce_last_block_filters=reduce_last_block_filters,
                    name=name)
  return model
