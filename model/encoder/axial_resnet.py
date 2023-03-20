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

"""Implements Axial-ResNets proposed in Axial-DeepLab [1].

[1] Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
    ECCV 2020 Spotlight.
      Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
      Liang-Chieh Chen.
"""

import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import activations
from deeplab2.model.layers import axial_block_groups
from deeplab2.model.layers import convolutions
from deeplab2.model.layers import resized_fuse
from deeplab2.model.layers import stems

# Add a suffix in layer names that indicate if the current layer is a part of
# the backbone or an extra layer, i.e. if the current layer will be pretrained
# or not. This name will be used when we apply 10x larger learning rates for
# extra parameters that have not been pretrained, in panoptic segmentation.
# This keyword is reserved and should not be a part of the variable names in a
# classification pretrained backbone.
EXTRA = 'extra'
# Similarly, we will apply 10x larger learning rates on the memory feature.
# This global variable name will be accessed when we build the optimizers. This
# keyword is reserved and should not be a part of the variable names in a
# classification pretrained backbone.
MEMORY_FEATURE = 'memory_feature'


class AxialResNet(tf.keras.Model):
  """An Axial-ResNet model as proposed in Axial-DeepLab [1] and MaX-DeepLab [2].

  An Axial-ResNet [1] replaces 3x3 convolutions in a Resnet by axial-attention
  layers. A dual-path transformer [2] and a stacked decoder [2] can be used
  optionally. In addition, this class supports scaling models with SWideRNet [3]
  and augmenting convolutions with Switchable Atrous Convolution [4].

  Reference:
  [1] Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
        Liang-Chieh Chen.
  [2] MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  [3] Scaling Wide Residual Networks for Panoptic Segmentation,
      https://arxiv.org/abs/2011.11675
        Liang-Chieh Chen, Huiyu Wang, Siyuan Qiao.
  [4] DetectoRS: Detecting Objects with Recursive Feature Pyramid and Switchable
      Atrous Convolution, CVPR 2021. https://arxiv.org/abs/2006.02334
        Siyuan Qiao, Liang-Chieh Chen, Alan Yuille.
  """

  def __init__(self,
               name,
               num_blocks=(3, 4, 6, 3),
               backbone_layer_multiplier=1.0,
               width_multiplier=1.0,
               stem_width_multiplier=1.0,
               output_stride=16,
               classification_mode=False,
               backbone_type='resnet_beta',
               use_axial_beyond_stride=16,
               backbone_use_transformer_beyond_stride=32,
               extra_decoder_use_transformer_beyond_stride=32,
               backbone_decoder_num_stacks=0,
               backbone_decoder_blocks_per_stage=1,
               extra_decoder_num_stacks=0,
               extra_decoder_blocks_per_stage=1,
               max_num_mask_slots=128,
               num_mask_slots=128,
               memory_channels=256,
               base_transformer_expansion=1.0,
               global_feed_forward_network_channels=256,
               high_resolution_output_stride=4,
               activation='relu',
               block_group_config=None,
               bn_layer=tf.keras.layers.BatchNormalization,
               conv_kernel_weight_decay=0.0):
    """Initializes an AxialResNet model.

    Args:
      name: A string, the name of the model.
      num_blocks: A list of 4 integers. It denotes the number of blocks to
        include in the last 4 stages or block groups. Each group consists of
        blocks that output features of the same resolution. Defaults to (3, 4,
        6, 3) as in MaX-DeepLab-S.
      backbone_layer_multiplier: A float, layer_multiplier for the backbone,
        excluding the STEM. This flag controls the number of layers. Defaults to
        1.0 as in MaX-DeepLab-S.
      width_multiplier: A float, the channel multiplier for the block groups.
        Defaults to 1.0 as in MaX-DeepLab-S.
      stem_width_multiplier: A float, the channel multiplier for stem
        convolutions. Defaults to 1.0 as in MaX-DeepLab-S.
      output_stride: An integer, the maximum ratio of input to output spatial
        resolution. Defaults to 16 as in MaX-DeepLab-S.
      classification_mode: A boolean, whether to perform in a classification
        mode. If it is True, this function directly returns backbone feature
        endpoints. Note that these feature endpoints can also be used directly
        for Panoptic-DeepLab or Motion-DeepLab. If it is False, this function
        builds MaX-DeepLab extra decoder layers and extra transformer layers.
        Defaults to False as in MaX-DeepLab.
      backbone_type: A string, the type of backbone. Supports 'resnet',
        'resnet_beta', and 'wider_resnet'. It controls both the stem type and
        the residual block type. Defaults to 'resnet_beta' as in MaX-DeepLab-S.
      use_axial_beyond_stride: An integer, the stride beyond which we use axial
        attention. Set to 0 if no axial attention is desired. Defaults to 16 as
        in MaX-DeepLab.
      backbone_use_transformer_beyond_stride: An integer, the stride beyond
        which we use a memory path transformer block on top of a regular pixel
        path block, in the backbone. Set to 0 if no transformer block is desired
        in the backbone. Defaults to 32 as in MaX-DeepLab-S.
      extra_decoder_use_transformer_beyond_stride: An integer, the stride beyond
        which we use a memory path transformer block on top of a regular pixel
        path block, in the extra decoder stages. Set to 0 if no transformer
        block is desired in the extra decoder stages. Defaults to 32 as in
        MaX-DeepLab-S.
      backbone_decoder_num_stacks: An integer, the number of decoder stacks
        (introduced in MaX-DeepLab) that we use in the backbone. The stacked
        decoders are applied in a stacked hour-glass style. Defaults to 0 as in
        MaX-DeepLab-S.
      backbone_decoder_blocks_per_stage: An integer, the number of consecutive
        residual blocks to apply for each decoder stage, in the backbone.
        Defaults to 1 as in MaX-DeepLab-S.
      extra_decoder_num_stacks: An integer, the number of decoder stacks
        (introduced in MaX-DeepLab) that we use in the extra decoder layers. It
        is different from backbone_decoder_blocks_per_stage in that the extra
        decoder stacks will be trained from scratch on segmentation tasks,
        instead of pretrained on ImageNet classification. Defaults to 0 as in
        MaX-DeepLab-S.
      extra_decoder_blocks_per_stage: An integer, the number of consecutive
        residual blocks to apply for each decoder stage, in the extra decoder
        stages. Defaults to 1 as in MaX-DeepLab-S.
      max_num_mask_slots: An integer, the maximum possible number of mask slots
        that will be used. This will be used in a pretraining-finetuning use
        case with different num_mask_slots: We can set max_num_mask_slots to the
        maximum possible num_mask_slots, and then the saved checkpoint can be
        loaded for finetuning with a different num_mask_slots. Defaults to 128
        as in MaX-DeepLab.
      num_mask_slots: An integer, the number of mask slots that will be used.
        Defaults to 128 as in MaX-DeepLab-S.
      memory_channels: An integer, the number of channels for the whole memory
        path. Defaults to 256 as in MaX-DeepLab-S.
      base_transformer_expansion: A float, the base width expansion rate for
        transformer layers. Defaults to 1.0 as in MaX-DeepLab-S.
      global_feed_forward_network_channels: An integer, the number of channels
        in the final global feed forward network, i.e. the mask feature head and
        the mask class head. Defaults to 256 as in MaX-DeepLab-S.
      high_resolution_output_stride: An integer, the final decoding output
        stride. Defaults to 4 as in MaX-DeepLab-S.
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.
      block_group_config: An argument dictionary that will be passed to
        block_group.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.

    Raises:
      ValueError: If backbone_type is not one of 'resnet', 'resnet_beta', or
        'wider_resnet'.
      ValueError: If extra_decoder_blocks_per_stage is not greater than zero.
    """
    super(AxialResNet, self).__init__(name=name)

    if extra_decoder_blocks_per_stage <= 0:
      raise ValueError(
          'Extra_decoder_blocks_per_stage should be great than zero.')
    if block_group_config is None:
      block_group_config = {}

    # Compute parameter lists for block_groups. We consider five stages so that
    # it is general enough to cover fully axial resnets and wider resnets.
    total_strides_list = [1, 2, 4, 8, 16]

    # Append 3 blocks for the first stage of fully axial resnets and wider
    # resnets.
    num_blocks_list = [3] + utils.scale_int_list(list(num_blocks),
                                                 backbone_layer_multiplier)
    strides_list = [2] * 5

    # Expand the transformer and the block filters with the stride.
    transformer_expansions_list = []
    filters_list = []
    for index, stride in enumerate(total_strides_list):
      # Reduce the number of channels when we apply transformer to low level
      # features (stride = 2, 4, or 8). The base_transformer_expansion is used
      # for stride = 16, i.e. the standard output_stride for MaX-DeepLab-S.
      transformer_expansions_list.append(base_transformer_expansion * stride /
                                         16.0)
      # Compute the base number of filters in each stage. For example, the last
      # stage of ResNet50 has an input stride of 16, then we compute the base
      # number of filters for a bottleneck block as 16 * 32 = 512, which is the
      # number of filters for the 3x3 convolution in those blocks.
      if backbone_type == 'wider_resnet' and index == 0:
        # SWideRNet variants use stem_width_multiplier for the first block.
        filters_list.append(int(round(stride * 32 * stem_width_multiplier)))
      else:
        filters_list.append(int(round(stride * 32 * width_multiplier)))

    self._num_mask_slots = None
    # Initialize memory_feature only when a transformer block is used.
    self._use_memory_feature = (backbone_use_transformer_beyond_stride or
                                (extra_decoder_use_transformer_beyond_stride and
                                 (not classification_mode)))
    if self._use_memory_feature:
      self._memory_feature_shape = (1, max_num_mask_slots, memory_channels)
      self._memory_feature_initializer = (
          tf.keras.initializers.TruncatedNormal(stddev=1.0))
      self._memory_feature_regularizer = tf.keras.regularizers.l2(
          conv_kernel_weight_decay)
      if num_mask_slots:
        self._num_mask_slots = num_mask_slots

    # Use a convolutional stem except fully axial cases.
    stem_channels = int(round(64 * stem_width_multiplier))
    self._activation_fn = activations.get_activation(activation)
    if use_axial_beyond_stride == 1:
      self._stem = tf.identity
      first_block_index = 0
    elif backbone_type.lower() == 'wider_resnet':
      self._stem = convolutions.Conv2DSame(
          output_channels=stem_channels,
          kernel_size=3,
          name='stem',
          strides=2,
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation='none',
          conv_kernel_weight_decay=conv_kernel_weight_decay)
      # Wider ResNet has five residual block stages, so we start from index 0.
      first_block_index = 0
      # Since we have applied the first strided convolution here, we do not use
      # a stride for the first stage (which will operate on stride 2).
      strides_list[0] = 1
      total_strides_list[0] = 2
    elif backbone_type.lower() == 'resnet_beta':
      self._stem = stems.InceptionSTEM(
          bn_layer=bn_layer,
          width_multiplier=stem_width_multiplier,
          conv_kernel_weight_decay=conv_kernel_weight_decay,
          activation=activation)
      first_block_index = 1
    elif backbone_type.lower() == 'resnet':
      self._stem = convolutions.Conv2DSame(
          output_channels=stem_channels,
          kernel_size=7,
          name='stem',
          strides=2,
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation='none',
          conv_kernel_weight_decay=conv_kernel_weight_decay)
      first_block_index = 1
    else:
      raise ValueError(backbone_type + ' is not supported.')

    self._first_block_index = first_block_index
    # Apply standard ResNet block groups. We use first_block_index to
    # distinguish models with 4 stages and those with 5 stages.
    for index in range(first_block_index, 5):
      current_name = '_stage{}'.format(index + 1)
      utils.safe_setattr(self, current_name, axial_block_groups.BlockGroup(
          filters=filters_list[index],
          num_blocks=num_blocks_list[index],
          name=utils.get_layer_name(current_name),
          original_resnet_stride=strides_list[index],
          original_resnet_input_stride=total_strides_list[index],
          output_stride=output_stride,
          backbone_type=backbone_type,
          use_axial_beyond_stride=use_axial_beyond_stride,
          use_transformer_beyond_stride=(
              backbone_use_transformer_beyond_stride),
          transformer_expansion=transformer_expansions_list[index],
          activation=activation,
          bn_layer=bn_layer,
          conv_kernel_weight_decay=conv_kernel_weight_decay,
          **block_group_config))
    self._backbone_decoder_num_stacks = backbone_decoder_num_stacks
    self._classification_mode = classification_mode
    self._extra_decoder_num_stacks = extra_decoder_num_stacks
    self._output_stride = output_stride
    self._high_resolution_output_stride = high_resolution_output_stride
    self._width_multiplier = width_multiplier
    self._activation = activation
    self._bn_layer = bn_layer
    self._conv_kernel_weight_decay = conv_kernel_weight_decay
    self._backbone_use_transformer_beyond_stride = (
        backbone_use_transformer_beyond_stride)
    self._extra_decoder_use_transformer_beyond_stride = (
        extra_decoder_use_transformer_beyond_stride)

    # Keep track of the current stack so that we know when to stop.
    current_stack = 0
    # Track whether we are building the backbone. This will affect the backbone
    # related arguments, local learning rate, and so on.
    current_is_backbone = True

    if backbone_decoder_num_stacks == 0:
      # No stacked decoder is used in the backbone, so we have finished building
      # the backbone. We either return the classification endpoints, or continue
      # building a non-backbone decoder for panoptic segmentation.
      if self._classification_mode:
        return
      else:
        current_is_backbone = False
    if not current_is_backbone:
      # Now that we have finished building the backbone and no stacked decoder
      # is used in the backbone, so we start to build extra (i.e., non-backbone)
      # layers for panoptic segmentation.
      current_name = '_stage5_' + EXTRA
      utils.safe_setattr(
          self, current_name, axial_block_groups.BlockGroup(
              filters=filters_list[-1],
              num_blocks=extra_decoder_blocks_per_stage,
              name=utils.get_layer_name(current_name),
              original_resnet_stride=1,
              original_resnet_input_stride=32,
              output_stride=output_stride,
              backbone_type=backbone_type,
              use_axial_beyond_stride=use_axial_beyond_stride,
              use_transformer_beyond_stride=(
                  extra_decoder_use_transformer_beyond_stride),
              transformer_expansion=base_transformer_expansion,
              activation=activation,
              bn_layer=bn_layer,
              conv_kernel_weight_decay=conv_kernel_weight_decay,
              **block_group_config))

    # Compute parameter lists for stacked decoder.
    total_decoder_num_stacks = (
        backbone_decoder_num_stacks + extra_decoder_num_stacks)

    # Use a function to compute the next stride.
    next_stride_fn = lambda x: x // 2
    current_decoder_stride = output_stride
    decoder_stage = 0

    # Exit if we have enough stacks and reach the decoding output stride.
    while (current_stack < total_decoder_num_stacks or
           current_decoder_stride > high_resolution_output_stride):
      decoder_stage += 1
      current_decoder_stride = next_stride_fn(current_decoder_stride)

      if current_decoder_stride == output_stride:
        current_stack += 1
        # Always use blocks from the last resnet stage if the current stride is
        # output stride (the largest stride).
        original_resnet_input_stride = 32

        # Switch the decoder direction if we reach the largest stride.
        next_stride_fn = lambda x: x // 2
      else:
        original_resnet_input_stride = current_decoder_stride

      # Scale channels according to the strides.
      decoder_channels = original_resnet_input_stride * 64 * width_multiplier
      current_transformer_expansion = (
          base_transformer_expansion * current_decoder_stride / 16.0)

      # Apply a decoder block group for building the backbone.
      if current_is_backbone:
        current_name = '_decoder_stage{}'.format(decoder_stage)
        utils.safe_setattr(
            self, current_name, axial_block_groups.BlockGroup(
                filters=decoder_channels // 4,
                num_blocks=backbone_decoder_blocks_per_stage,
                name=utils.get_layer_name(current_name),
                original_resnet_stride=1,
                original_resnet_input_stride=original_resnet_input_stride,
                output_stride=output_stride,
                backbone_type=backbone_type,
                use_axial_beyond_stride=use_axial_beyond_stride,
                use_transformer_beyond_stride=(
                    backbone_use_transformer_beyond_stride),
                transformer_expansion=current_transformer_expansion,
                activation=activation,
                bn_layer=bn_layer,
                conv_kernel_weight_decay=conv_kernel_weight_decay,
                **block_group_config))

      if (current_decoder_stride == output_stride and
          current_stack == backbone_decoder_num_stacks):
        # Now that we have finished building the backbone, we either return the
        # classification endpoints, or continue building a non-backbone decoder
        # for panoptic segmentation.
        if classification_mode:
          return
        else:
          current_is_backbone = False

      # Apply a decoder block group for building the extra layers.
      if not current_is_backbone:
        # Continue building an extra (i.e., non-backbone) decoder for panoptic
        # segmentation.
        current_name = '_decoder_stage{}_{}'.format(decoder_stage, EXTRA)
        utils.safe_setattr(
            self, current_name, axial_block_groups.BlockGroup(
                filters=decoder_channels // 4,
                num_blocks=extra_decoder_blocks_per_stage,
                name=utils.get_layer_name(current_name),
                original_resnet_stride=1,
                original_resnet_input_stride=original_resnet_input_stride,
                output_stride=output_stride,
                backbone_type=backbone_type,
                use_axial_beyond_stride=use_axial_beyond_stride,
                use_transformer_beyond_stride=(
                    extra_decoder_use_transformer_beyond_stride),
                transformer_expansion=current_transformer_expansion,
                activation=activation,
                bn_layer=bn_layer,
                conv_kernel_weight_decay=conv_kernel_weight_decay,
                **block_group_config))
      if current_decoder_stride == high_resolution_output_stride:
        next_stride_fn = lambda x: x * 2

    # Assert that we have already returned if we are building a classifier.
    assert not classification_mode
    if (backbone_use_transformer_beyond_stride or
        extra_decoder_use_transformer_beyond_stride):
      # Build extra memory path feed forward networks for the class feature and
      # the mask feature.
      current_name = '_class_feature_' + EXTRA
      utils.safe_setattr(
          self, current_name, convolutions.Conv1D(
              global_feed_forward_network_channels,
              utils.get_layer_name(current_name),
              use_bias=False,
              use_bn=True,
              bn_layer=bn_layer,
              activation=activation,
              conv_kernel_weight_decay=conv_kernel_weight_decay))
      current_name = '_mask_feature_' + EXTRA
      utils.safe_setattr(
          self, current_name, convolutions.Conv1D(
              global_feed_forward_network_channels,
              utils.get_layer_name(current_name),
              use_bias=False,
              use_bn=True,
              bn_layer=bn_layer,
              activation=activation,
              conv_kernel_weight_decay=conv_kernel_weight_decay))

  def build(self, input_shape):
    """Builds model weights and input shape dependent sub-layers."""
    if self._use_memory_feature:
      self._memory_feature = self.add_weight(
          name=MEMORY_FEATURE,
          shape=self._memory_feature_shape,
          initializer=self._memory_feature_initializer,
          regularizer=self._memory_feature_regularizer)
    else:
      self._memory_feature = None

    # Go through the loop to build the ResizedFuse layers.
    current_stack = 0
    # Track whether we are building the backbone. This will affect the backbone
    # related arguments, local learning rate, and so on.
    current_is_backbone = self._backbone_decoder_num_stacks != 0
    total_decoder_num_stacks = (
        self._backbone_decoder_num_stacks + self._extra_decoder_num_stacks)
    next_stride_fn = lambda x: x // 2
    current_decoder_stride = self._output_stride
    decoder_stage = 0
    while (current_stack < total_decoder_num_stacks or
           current_decoder_stride > self._high_resolution_output_stride):
      decoder_stage += 1
      current_decoder_stride = next_stride_fn(current_decoder_stride)
      if current_decoder_stride == self._output_stride:
        current_stack += 1
        original_resnet_input_stride = 32
        next_stride_fn = lambda x: x // 2
      else:
        original_resnet_input_stride = current_decoder_stride
      # Compute the decoder_channels according to original_resnet_input_stride.
      # For example, at stride 4 with width multiplier = 1, we use 4 * 64 = 256
      # channels, which is the same as a standard ResNet.
      decoder_channels = int(round(
          original_resnet_input_stride * 64 * self._width_multiplier))
      decoder_height, decoder_width = utils.scale_mutable_sequence(
          input_shape[1:3], 1.0 / current_decoder_stride)
      if current_is_backbone:
        current_name = '_decoder_stage{}_resized_fuse'.format(decoder_stage)
      else:
        current_name = '_decoder_stage{}_{}_resized_fuse'.format(
            decoder_stage, EXTRA)
      utils.safe_setattr(
          self, current_name, resized_fuse.ResizedFuse(
              name=utils.get_layer_name(current_name),
              height=decoder_height,
              width=decoder_width,
              num_channels=decoder_channels,
              activation=self._activation,
              bn_layer=self._bn_layer,
              conv_kernel_weight_decay=self._conv_kernel_weight_decay))
      if (current_decoder_stride == self._output_stride and
          current_stack == self._backbone_decoder_num_stacks):
        # Now that we have finished building the backbone, we either return the
        # classification endpoints, or continue building a non-backbone decoder
        # for panoptic segmentation.
        if self._classification_mode:
          return
        current_is_backbone = False
      if current_decoder_stride == self._high_resolution_output_stride:
        next_stride_fn = lambda x: x * 2

  def call_encoder_before_stacked_decoder(self, inputs, training=False):
    """Performs a forward pass of the encoder before stacking decoders.

    Args:
      inputs: An input [batch, height, width, channel] tensor.
      training: A boolean, whether the model is in training mode.

    Returns:
      current_output: An output tensor with shape [batch, new_height, new_width,
        new_channel].
      activated_output: An activated output tensor with shape [batch,
        new_height, new_width, new_channel].
      memory_feature: None if no transformer is used. A [batch, num_memory,
        memory_channel] tensor if transformer is used.
      endpoints: A dict, the network endpoints that might be used by DeepLab.
    """
    memory_feature = self._memory_feature
    if self._use_memory_feature:
      if self._num_mask_slots:
        memory_feature = self._memory_feature[:, :self._num_mask_slots, :]
      memory_feature = tf.tile(memory_feature,
                               [tf.shape(inputs)[0], 1, 1])

    endpoints = {}
    current_output = self._stem(inputs)
    activated_output = self._activation_fn(current_output)
    endpoints['stage1'] = current_output
    endpoints['res1'] = activated_output

    # Apply standard ResNet block groups. We use first_block_index to
    # distinguish models with 4 stages and those with 5 stages.
    for index in range(self._first_block_index, 5):
      current_name = '_stage{}'.format(index + 1)
      current_output, memory_feature = (
          getattr(self, current_name)(
              (current_output, memory_feature), training=training))
      activated_output = self._activation_fn(current_output)
      endpoints[utils.get_layer_name(current_name)] = current_output
      activated_output_name = 'res{}'.format(index + 1)
      endpoints[activated_output_name] = activated_output
    return current_output, activated_output, memory_feature, endpoints

  def call_stacked_decoder(self,
                           current_output,
                           activated_output,
                           memory_feature,
                           endpoints,
                           training=False):
    """Performs a forward pass of the stacked decoders.

    Args:
      current_output: An output tensor with shape [batch, new_height, new_width,
        new_channel].
      activated_output: An activated output tensor with shape [batch,
        new_height, new_width, new_channel].
      memory_feature: None if no transformer is used. A [batch, num_memory,
        memory_channel] tensor if transformer is used.
      endpoints: A dict, the network endpoints that might be used by DeepLab.
      training: A boolean, whether the model is in training mode.

    Returns:
      memory_feature: None if no transformer is used. A [batch, num_memory,
        memory_channel] tensor if transformer is used.
      high_resolution_outputs: A list of decoded tensors with
        high_resolution_output_stride.
      backbone_output: An output tensor of the backbone, with output_stride.
      endpoints: A dict, the network endpoints that might be used by DeepLab.
    """
    # Keep track of the current stack so that we know when to stop.
    current_stack = 0
    # Track whether we are building the backbone. This will affect the backbone
    # related arguments, local learning rate, and so on.
    current_is_backbone = True
    high_resolution_outputs = []

    if self._backbone_decoder_num_stacks == 0:
      # Keep track of the backbone output, since it might be used as the
      # semantic feature output.
      backbone_output = activated_output
      # Now that we have finished building the backbone, we either return the
      # classification logits, or continue building a non-backbone decoder for
      # panoptic segmentation.
      if self._classification_mode:
        endpoints['backbone_output'] = backbone_output
        return None, None, None, endpoints
      else:
        current_is_backbone = False

    if not current_is_backbone:
      # Build extra layers if we have finished building the backbone.
      current_name = '_stage5_' + EXTRA
      current_output, memory_feature = (
          getattr(self, current_name)(
              (current_output, memory_feature), training=training))

    # Compute parameter lists for stacked decoder.
    total_decoder_num_stacks = (
        self._backbone_decoder_num_stacks + self._extra_decoder_num_stacks)

    # Keep track of all endpoints that will be used in the stacked decoder.
    stride_to_features = {}
    stride_to_features[min(2, self._output_stride)] = [endpoints['stage1']]
    stride_to_features[min(4, self._output_stride)] = [endpoints['stage2']]
    stride_to_features[min(8, self._output_stride)] = [endpoints['stage3']]
    stride_to_features[min(16, self._output_stride)] = [endpoints['stage4']]
    # Only keep the last endpoint from the backbone with the same resolution,
    # i.e., if the output stride is 16, the current output will override
    # the stride 16 endpoint, endpoints['res4'].
    stride_to_features[min(32, self._output_stride)] = [current_output]

    # Use a function to compute the next stride.
    next_stride_fn = lambda x: x // 2
    current_decoder_stride = self._output_stride
    decoder_stage = 0

    # Exit if we have enough stacks and reach the decoding output stride.
    while (current_stack < total_decoder_num_stacks or
           current_decoder_stride > self._high_resolution_output_stride):
      decoder_stage += 1
      current_decoder_stride = next_stride_fn(current_decoder_stride)

      if current_decoder_stride == self._output_stride:
        current_stack += 1
        # Switch the decoder direction if we reach the largest stride.
        next_stride_fn = lambda x: x // 2

      # Include the current feature and two previous features from the target
      # resolution in the decoder. We select two because it contains one upward
      # feature and one downward feature, but better choices are possible.
      decoder_features_list = (
          [current_output] +
          stride_to_features[current_decoder_stride][-2:])

      # Fuse and resize features with striding, resizing and 1x1 convolutions.
      if current_is_backbone:
        current_name = '_decoder_stage{}_resized_fuse'.format(decoder_stage)
      else:
        current_name = '_decoder_stage{}_{}_resized_fuse'.format(
            decoder_stage, EXTRA)
      current_output = getattr(self, current_name)(
          decoder_features_list, training=training)

      # Apply a decoder block group for building the backbone.
      if current_is_backbone:
        current_name = '_decoder_stage{}'.format(decoder_stage)
        current_output, memory_feature = (
            getattr(self, current_name)(
                (current_output, memory_feature), training=training))

      if (current_decoder_stride == self._output_stride and
          current_stack == self._backbone_decoder_num_stacks):
        # Keep track of the backbone output, since it might be used as the
        # semantic feature output.
        activated_output = self._activation_fn(current_output)
        backbone_output = activated_output
        # Now that we have finished building the backbone, we either return the
        # classification logits, or continue building a non-backbone decoder for
        # panoptic segmentation.
        if self._classification_mode:
          endpoints['backbone_output'] = backbone_output
          return None, None, None, endpoints
        else:
          current_is_backbone = False

      # Apply a decoder block group for building the extra layers.
      if not current_is_backbone:
        current_name = '_decoder_stage{}_{}'.format(decoder_stage, EXTRA)
        current_output, memory_feature = (
            getattr(self, current_name)(
                (current_output, memory_feature), training=training))

      # Append the current feature into the feature dict for possible later
      # usage.
      stride_to_features[current_decoder_stride].append(current_output)
      if current_decoder_stride == self._high_resolution_output_stride:
        activated_output = self._activation_fn(current_output)
        high_resolution_outputs.append(activated_output)
        next_stride_fn = lambda x: x * 2
    return memory_feature, high_resolution_outputs, backbone_output, endpoints

  def call_extra_endpoints(self,
                           memory_feature,
                           high_resolution_outputs,
                           backbone_output,
                           endpoints,
                           training=False):
    """Performs a forward pass to generate extra endpoints.

    Args:
      memory_feature: None if no transformer is used. A [batch, num_memory,
        memory_channel] tensor if transformer is used.
      high_resolution_outputs: A list of decoded tensors with
        high_resolution_output_stride.
      backbone_output: An output tensor of the backbone, with output_stride.
      endpoints: A dict, the network endpoints that might be used by DeepLab.
      training: A boolean, whether the model is in training mode.

    Returns:
      endpoints: A dict, the network endpoints that might be used by DeepLab.
    """
    # Assert that we have already returned if we are building a classifier.
    assert not self._classification_mode
    if (self._backbone_use_transformer_beyond_stride or
        self._extra_decoder_use_transformer_beyond_stride):
      # Build extra memory path feed forward networks for the class feature and
      # the mask feature.
      class_feature = getattr(self, '_class_feature_' + EXTRA)(
          memory_feature, training=training)
      mask_feature = getattr(self, '_mask_feature_' + EXTRA)(
          memory_feature, training=training)
      endpoints['transformer_class_feature'] = class_feature
      endpoints['transformer_mask_feature'] = mask_feature

    # Output the last high resolution feature as panoptic feature.
    endpoints['feature_panoptic'] = high_resolution_outputs[-1]

    # Avoid sharing our panoptic feature with the semantic auxiliary loss. So we
    # use the backbone feature or the decoded backbone feature for the semantic
    # segmentation head (i.e. the auxiliary loss).
    if self._extra_decoder_num_stacks:
      endpoints['feature_semantic'] = (
          high_resolution_outputs[self._backbone_decoder_num_stacks])
    else:
      endpoints['feature_semantic'] = backbone_output
    endpoints['backbone_output'] = backbone_output
    return endpoints

  def call(self, inputs, training=False):
    """Performs a forward pass.

    Args:
      inputs: An input [batch, height, width, channel] tensor.
      training: A boolean, whether the model is in training mode.

    Returns:
      endpoints: A dict, the network endpoints that might be used by DeepLab.
    """
    current_output, activated_output, memory_feature, endpoints = (
        self.call_encoder_before_stacked_decoder(inputs, training=training))
    memory_feature, high_resolution_outputs, backbone_output, endpoints = (
        self.call_stacked_decoder(current_output,
                                  activated_output,
                                  memory_feature,
                                  endpoints,
                                  training=training))
    if self._classification_mode:
      return endpoints
    endpoints = self.call_extra_endpoints(memory_feature,
                                          high_resolution_outputs,
                                          backbone_output,
                                          endpoints,
                                          training=training)
    return endpoints
