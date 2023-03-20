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

"""Implements Axial-Blocks proposed in Axial-DeepLab [1].

Axial-Blocks are based on residual bottleneck blocks, but with the 3x3
convolution replaced with two axial-attention layers, one on the height-axis,
followed by the other on the width-axis.

[1] Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
    ECCV 2020 Spotlight.
      Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
      Liang-Chieh Chen.
"""
import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import activations
from deeplab2.model.layers import axial_layers
from deeplab2.model.layers import convolutions
from deeplab2.model.layers import squeeze_and_excite


class AxialBlock(tf.keras.layers.Layer):
  """An AxialBlock as a building block for an Axial-ResNet model.

  We implement the Axial-Block proposed in [1] in a general way that also
  includes convolutional residual blocks, such as the basic block and the
  bottleneck block (w/ and w/o Switchable Atrous Convolution).

  A basic block consists of two 3x3 convolutions and a residual connection. It
  is the main building block for wide-resnet variants.

  A bottleneck block consists of consecutive 1x1, 3x3, 1x1 convolutions and a
  residual connection. It is the main building block for standard resnet
  variants.

  An axial block consists of a 1x1 input convolution, a self-attention layer
  (either axial-attention or global attention), a 1x1 output convolution, and a
  residual connection. It is the main building block for axial-resnet variants.

  Note: We apply the striding in the first spatial operation (i.e. 3x3
  convolution or self-attention layer).
  """

  def __init__(self,
               filters_list,
               kernel_size=3,
               strides=1,
               atrous_rate=1,
               use_squeeze_and_excite=False,
               use_sac=False,
               bn_layer=tf.keras.layers.BatchNormalization,
               activation='relu',
               name=None,
               conv_kernel_weight_decay=0.0,
               basic_block_second_conv_atrous_rate=None,
               attention_type=None,
               axial_layer_config=None):
    """Initializes an AxialBlock.

    Args:
      filters_list: A list of filter numbers in the residual block. We currently
        support filters_list with two or three elements. Two elements specify
        the filters for two consecutive 3x3 convolutions, while three elements
        specify the filters for three convolutions (1x1, 3x3, and 1x1).
      kernel_size: The size of the convolution kernels (default: 3).
      strides: The strides of the block (default: 1).
      atrous_rate: The atrous rate of the 3x3 convolutions (default: 1). If this
        residual block is a basic block, it is recommendeded to specify correct
        basic_block_second_conv_atrous_rate for the second 3x3 convolution.
        Otherwise, the second conv will also use atrous rate, which might cause
        atrous inconsistency with different output strides, as tested in
        axial_block_groups_test.test_atrous_consistency_basic_block.
      use_squeeze_and_excite: A boolean flag indicating whether
        squeeze-and-excite (SE) is used.
      use_sac: A boolean, using the Switchable Atrous Convolution (SAC) or not.
      bn_layer: A tf.keras.layers.Layer that computes the normalization
        (default: tf.keras.layers.BatchNormalization).
      activation: A string specifying the activation function to apply.
      name: An string specifying the name of the layer (default: None).
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.
      basic_block_second_conv_atrous_rate: An integer, the atrous rate for the
        second convolution of basic block. This is necessary to ensure atrous
        consistency with different output_strides. Defaults to atrous_rate.
      attention_type: A string, type of attention to apply. Support 'axial' and
        'global'.
      axial_layer_config: A dict, an argument dictionary for the axial layer.

    Raises:
      ValueError: If filters_list does not have two or three elements.
      ValueError: If attention_type is not supported.
      ValueError: If double_global_attention is True in axial_layer_config.
    """
    super(AxialBlock, self).__init__(name=name)

    self._filters_list = filters_list
    self._strides = strides
    self._use_squeeze_and_excite = use_squeeze_and_excite
    self._bn_layer = bn_layer
    self._activate_fn = activations.get_activation(activation)
    self._attention_type = attention_type

    if axial_layer_config is None:
      axial_layer_config = {}

    if basic_block_second_conv_atrous_rate is None:
      basic_block_second_conv_atrous_rate = atrous_rate

    if len(filters_list) == 3:
      # Three consecutive convolutions: 1x1, 3x3, and 1x1.
      self._conv1_bn_act = convolutions.Conv2DSame(
          filters_list[0], 1, 'conv1_bn_act',
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation=activation,
          conv_kernel_weight_decay=conv_kernel_weight_decay)

      if attention_type is None or attention_type.lower() == 'none':
        self._conv2_bn_act = convolutions.Conv2DSame(
            filters_list[1], kernel_size, 'conv2_bn_act',
            strides=strides,
            atrous_rate=atrous_rate,
            use_bias=False,
            use_bn=True,
            bn_layer=bn_layer,
            activation=activation,
            use_switchable_atrous_conv=use_sac,
            # We default to use global context in SAC if use_sac is True. This
            # setting is experimentally found effective.
            use_global_context_in_sac=use_sac,
            conv_kernel_weight_decay=conv_kernel_weight_decay)
      elif attention_type == 'axial':
        if 'double_global_attention' in axial_layer_config:
          if axial_layer_config['double_global_attention']:
            raise ValueError('Double_global_attention takes no effect in '
                             'AxialAttention2D.')
          del axial_layer_config['double_global_attention']
        self._attention = axial_layers.AxialAttention2D(
            strides=strides,
            filters=filters_list[1],
            name='attention',
            bn_layer=bn_layer,
            conv_kernel_weight_decay=conv_kernel_weight_decay,
            **axial_layer_config)
      elif attention_type == 'global':
        self._attention = axial_layers.GlobalAttention2D(
            strides=strides,
            filters=filters_list[1],
            name='attention',
            bn_layer=bn_layer,
            conv_kernel_weight_decay=conv_kernel_weight_decay,
            **axial_layer_config)
      else:
        raise ValueError(attention_type + ' is not supported.')

      # Here we apply a batch norm with gamma initialized at zero. This ensures
      # that at random initialization of the model, the skip connections
      # dominate all residual blocks. In this way, all the skip connections
      # construct an identity mapping that passes the gradients (without any
      # distortion from the randomly initialized blocks) to all residual blocks.
      # This trick helps training at early epochs.
      # Reference: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour".
      # https://arxiv.org/abs/1706.02677
      self._conv3_bn = convolutions.Conv2DSame(
          filters_list[2], 1, 'conv3_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          bn_gamma_initializer='zeros',
          activation='none',
          conv_kernel_weight_decay=conv_kernel_weight_decay)
    elif len(filters_list) == 2:
      # Two consecutive convolutions: 3x3 and 3x3.
      self._conv1_bn_act = convolutions.Conv2DSame(
          filters_list[0], kernel_size, 'conv1_bn_act',
          strides=strides,
          atrous_rate=atrous_rate,
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation=activation,
          use_switchable_atrous_conv=use_sac,
          use_global_context_in_sac=use_sac,
          conv_kernel_weight_decay=conv_kernel_weight_decay)
      # Here we apply a batch norm with gamma initialized at zero. This ensures
      # that at random initialization of the model, the skip connections
      # dominate all residual blocks. In this way, all the skip connections
      # construct an identity mapping that passes the gradients (without any
      # distortion from the randomly initialized blocks) to all residual blocks.
      # This trick helps training at early epochs.
      # Reference: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour".
      # https://arxiv.org/abs/1706.02677
      self._conv2_bn = convolutions.Conv2DSame(
          filters_list[1], kernel_size, 'conv2_bn',
          strides=1,
          atrous_rate=basic_block_second_conv_atrous_rate,
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          bn_gamma_initializer='zeros',
          activation='none',
          use_switchable_atrous_conv=use_sac,
          use_global_context_in_sac=use_sac,
          conv_kernel_weight_decay=conv_kernel_weight_decay)
    else:
      raise ValueError('Expect filters_list to have length 2 or 3; got %d' %
                       len(filters_list))

    if self._use_squeeze_and_excite:
      self._squeeze_and_excite = squeeze_and_excite.SimplifiedSqueezeAndExcite(
          filters_list[-1])
    self._conv_kernel_weight_decay = conv_kernel_weight_decay

  def build(self, input_shape_list):
    input_tensor_shape = input_shape_list[0]
    self._shortcut = None
    if input_tensor_shape[3] != self._filters_list[-1]:
      self._shortcut = convolutions.Conv2DSame(
          self._filters_list[-1], 1, 'shortcut',
          strides=self._strides,
          use_bias=False,
          use_bn=True,
          bn_layer=self._bn_layer,
          activation='none',
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)

  def call(self, inputs):
    """Performs a forward pass.

    We have to define drop_path_random_mask outside the layer call and pass it
    into the layer, because recompute_grad (gradient checkpointing) does not
    allow any randomness within the function call. In addition, recompute_grad
    only supports float tensors as inputs. For this reason, the training flag
    should be also passed as a float tensor. For the same reason, we cannot
    support passing drop_path_random_mask as None. Instead, we ask the users to
    pass only the first two tensors when drop path is not used.

    Args:
      inputs: A tuple of 2 or 3 tensors, containing
        input_tensor should be an input tensor of type tf.Tensor with shape
          [batch, height, width, channels].
        float_tensor_training should be a float tensor of 0.0 or 1.0, whether
          the model is in training mode.
        (optional) drop_path_random_mask is a drop path random mask of type
          tf.Tensor with shape [batch, 1, 1, 1].

    Returns:
      outputs: A single tensor without activation function.

    Raises:
      ValueError: If the length of inputs is not 2 or 3.
    """
    if len(inputs) not in (2, 3):
      raise ValueError('The length of inputs should be either 2 or 3.')

    # Unpack the inputs.
    input_tensor, float_tensor_training, drop_path_random_mask = (
        utils.pad_sequence_with_none(inputs, target_length=3))

    # AxialBlock expects an activated feature as input_tensor.
    input_tensor = self._activate_fn(input_tensor)

    # Recompute_grad takes only float tensors as inputs. It does not allow
    # bools or boolean tensors. For this reason, we cast training to a float
    # tensor outside this call, and now we cast it back to a boolean tensor.
    training = tf.cast(float_tensor_training, tf.bool)

    shortcut = input_tensor
    if self._shortcut is not None:
      shortcut = self._shortcut(shortcut, training=training)
    elif self._strides != 1:
      shortcut = shortcut[:, ::self._strides, ::self._strides, :]

    if len(self._filters_list) == 3:
      x = self._conv1_bn_act(input_tensor, training=training)
      if (self._attention_type is None or
          self._attention_type.lower() == 'none'):
        x = self._conv2_bn_act(x, training=training)
      else:
        x = self._attention(x, training=training)
        x = self._activate_fn(x)
      x = self._conv3_bn(x, training=training)
    if len(self._filters_list) == 2:
      x = self._conv1_bn_act(input_tensor, training=training)
      x = self._conv2_bn(x, training=training)

    if self._use_squeeze_and_excite:
      x = self._squeeze_and_excite(x)

    if drop_path_random_mask is not None:
      x = x * drop_path_random_mask
    x = x + shortcut
    return x
