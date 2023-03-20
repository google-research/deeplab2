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

"""Implements convolutional and attentional residual block groups."""

import math
import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import activations
from deeplab2.model.layers import axial_blocks
from deeplab2.model.layers import drop_path
from deeplab2.model.layers import dual_path_transformer
from deeplab2.model.layers import positional_encodings
from deeplab2.model.layers import recompute_grad as recompute_grad_lib

# We will apply 10x larger learning rates on transformer layers. This global
# variable name will be accessed when we build the optimizers. This keyword is
# reserved and should not be a part of the variable names in a classification
# pretrained backbone.
TRANSFORMER = 'transformer'


def _get_current_names(index):
  current_name = '_block{}'.format(index + 1)
  transformer_current_name = '_block{}_{}'.format(index + 1, TRANSFORMER)
  return current_name, transformer_current_name


class BlockGroup(tf.keras.layers.Layer):
  """Applies a group of residual blocks with dual path transformer layers [1].

  An optional dual-path transformer layer is inserted after each residual block.
  The transformer layer performs memory2pixel attention, pixel2memory attention,
  and memory2memory self-attention, while the standard residual block applies
  the pixel2pixel axial-attention, global-attention, or spatial convolution.

  Reference:
  [1] MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  def __init__(self,
               filters,
               num_blocks,
               name,
               original_resnet_stride,
               original_resnet_input_stride,
               output_stride=16,
               backbone_type='resnet_beta',
               positional_encoding_type=None,
               use_global_beyond_stride=0,
               use_axial_beyond_stride=16,
               use_transformer_beyond_stride=32,
               use_sac_beyond_stride=0,
               use_squeeze_and_excite=False,
               conv_use_recompute_grad=False,
               axial_use_recompute_grad=True,
               recompute_within_stride=0,
               transformer_use_recompute_grad=False,
               transformer_expansion=1,
               drop_path_keep_prob=0.8,
               drop_path_beyond_stride=16,
               drop_path_schedule='constant',
               activation='relu',
               attention_bottleneck_expansion=2,
               axial_layer_config=None,
               dual_path_transformer_layer_config=None,
               bn_layer=tf.keras.layers.BatchNormalization,
               conv_kernel_weight_decay=0.0,
               auxiliary_predictor_func=None,
               use_axial_block=True):
    """Initializes a BlockGroup layer.

    Args:
      filters: An integer, the base number of channels for this block group.
      num_blocks: An integer, the number of blocks for this block group.
      name: A string, the name of the block group.
      original_resnet_stride: An integer, the original resnet stride for this
        block, usually 1 or 2. The stride will be applied if
        original_resnet_input_stride is smaller than the desired output_stride.
        Otherwise, the stride will not be applied, and atrous convolution will
        be used after the first block.
      original_resnet_input_stride: An integer, the total input stride in the
        original resnet. For example, the total input stride for the last stage
        of the original resnet is 16, and the total output stride is 32. This
        stride differs from the true stride of the feature in that we might use
        atrous convolution to change both the input and output stride to, e.g.
        8, but its original resnet input stride remains the same. In this case,
        we also use the original resnet input stride to compute the atrous rate.
      output_stride: An integer, the desired output_stride for the ResNet.
      backbone_type: A string, the type of the backbone. Supports 'resnet',
        'resnet_beta', and 'wider_resnet'. The 'resnet' refers to the original
        resnet with a 7x7 convolutional stem. The 'resnet_beta' means a resnet
        but with an inception stem. The 'wider_resnet' is a wider variant of
        resnet with extensively used 3x3 convolutions.
      positional_encoding_type: A string, type of the positional encoding.
        Support '2D', '1D', and None.
      use_global_beyond_stride: An integer, the stride beyond which we use
        global attention. Set to 0 if no global attention is desired. Defaults
        to 0, i.e. we do not use global attention.
      use_axial_beyond_stride: An integer, the stride beyond which we use axial
        attention. Note that use_global_beyond_stride has a higher priority,
        i.e. we use global attention if the stride is also beyond
        use_global_beyond_stride. Set to 0 if no axial attention is desired.
        Defaults to 16 as in MaX-DeepLab.
      use_transformer_beyond_stride: An integer, the stride beyond which we use
        a transformer layer. Set to 0 if no transformer is desired. Defaults to
        32 as in MaX-DeepLab-S.
      use_sac_beyond_stride: An integer. Use the Switchable Atrous Convolution
        (SAC) beyond the specified stride. For example, if
        `use_sac_beyond_stride` = 16, SAC will be applied to the network stage
        whose output stride >= 16 (i.e., 16 and 32). Set to 0 or -1 to disable
        it. Defaults to 0 as SAC is not used in MaX-DeepLab.
      use_squeeze_and_excite: A boolean, whether squeeze-and-excite (SE) is
        used. Defaults to False as SE is not used in MaX-DeepLab.
      conv_use_recompute_grad: A boolean, whether to use the gradient
        checkpointing trick for convolutional blocks. This trick reduces
        accelerator memory usage, but takes longer to compute gradients.
        Defaults to False since convolutional layers are memory efficient.
      axial_use_recompute_grad: A boolean, whether to use the gradient
        checkpointing trick for axial blocks. This trick reduces accelerator
        memory usage, but takes longer to compute gradients. Defaults to True
        since it saves memory for axial blocks.
      recompute_within_stride: An integer, the stride within which we use the
        gradient checkpointing trick. This trick reduces accelerator memory
        usage, but takes longer to compute gradients. Defaults to 0 (do not
        recompute any layer).
      transformer_use_recompute_grad: A boolean, whether to use the gradient
        checkpointing trick for dual-path transformer blocks. This trick reduces
        accelerator memory usage, but takes longer to compute gradients.
        Defaults to False.
      transformer_expansion: An integer, the expansion ratio for the transformer
        bottleneck.
      drop_path_keep_prob: A float, the keep probability for dropping path.
        Defaults to 0.8 as in MaX-DeepLab-S.
      drop_path_beyond_stride: An integer, the stride beyond which we apply drop
        path augmentation. Defaults to 16 as in MaX-DeepLab-S.
      drop_path_schedule: A string, the drop path schedule. Currently, we
        support 'constant': use the same drop path keep probability for all
        stages, and 'linear': linearly decrease the drop path keep probability
        from 1.0 at 0-th stage (or STEM) to `drop_path_keep_prob` at last stage.
      activation: A string, type of activation function to apply. Support
        'relu', 'swish' (or 'silu'), 'gelu', 'approximated_gelu', and 'elu'.
      attention_bottleneck_expansion: An integer, the expansion ratio for
        axial attention blocks.
      axial_layer_config: A dict, an argument dictionary for the axial layer.
      dual_path_transformer_layer_config: A dict, an argument dictionary for the
        transformer.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.
      auxiliary_predictor_func: A callable function that returns an
        initialization of auxiliary predictor.
      use_axial_block: A boolean. When set to False, the axial block will be
        disabled. This is helpful when we just want to perform
        dual-path transformer (i.e., no pixel2pixel attention). Default to True.

    Raises:
      ValueError: If backbone_type is not one of 'resnet', 'resnet_beta', or
        'wider_resnet'.
      ValueError: original_resnet_input_stride is not power of 2.
      ValueError: output_stride is not power of 2.
    """
    if original_resnet_input_stride & (original_resnet_input_stride - 1):
      raise ValueError('original_resnet_input_stride is not power of 2.')
    if output_stride & (output_stride - 1):
      raise ValueError('output_stride is not power of 2.')

    super(BlockGroup, self).__init__(name=name)
    self._add_absolute_positional_encoding = None
    self._activation_fn = activations.get_activation(activation)
    self._num_blocks = num_blocks
    self._drop_path_keep_prob = []
    self._recompute_grad = []
    self._transformer_use_recompute_grad = transformer_use_recompute_grad
    self._use_axial_block = use_axial_block
    if dual_path_transformer_layer_config is None:
      dual_path_transformer_layer_config = {}
    original_resnet_current_stride = original_resnet_input_stride

    use_sac = (original_resnet_input_stride * original_resnet_stride >=
               use_sac_beyond_stride > 0)

    recompute_grad = (original_resnet_input_stride * original_resnet_stride <=
                      recompute_within_stride)

    for index in range(num_blocks):
      current_name, transformer_current_name = _get_current_names(index)

      # Compute the current strides. If there is a stride for this block group,
      # we do it in the first residual block.
      if index == 0 and original_resnet_input_stride < output_stride:
        current_strides = original_resnet_stride
      else:
        current_strides = 1

      # Compute the current atrous rate.
      if original_resnet_current_stride > output_stride:
        atrous_rate = original_resnet_current_stride // output_stride
      else:
        atrous_rate = 1

      # Compute the atrous rate for the second conv in the first basic block.
      if (index == 0 and original_resnet_input_stride * original_resnet_stride >
          output_stride):
        basic_block_second_conv_atrous_rate = (
            original_resnet_input_stride * original_resnet_stride //
            output_stride)
      else:
        basic_block_second_conv_atrous_rate = atrous_rate

      # Compute the current drop_path_keep_prob.
      current_stage = math.log2(original_resnet_current_stride) - 1
      if original_resnet_current_stride >= drop_path_beyond_stride:
        current_drop_path_keep_prob = drop_path.get_drop_path_keep_prob(
            drop_path_keep_prob, drop_path_schedule,
            current_stage=int(round(current_stage)),
            num_stages=4)
      else:
        current_drop_path_keep_prob = 1.0

      # Compute which block_fn to use for this residual block.
      if original_resnet_current_stride >= use_global_beyond_stride > 0:
        attention_type = 'global'
        recompute_grad = axial_use_recompute_grad or recompute_grad
        filters_list = [filters * attention_bottleneck_expansion,
                        filters,
                        filters * 4]
      elif original_resnet_current_stride >= use_axial_beyond_stride > 0:
        attention_type = 'axial'
        recompute_grad = axial_use_recompute_grad or recompute_grad
        filters_list = [filters * attention_bottleneck_expansion,
                        filters,
                        filters * 4]
      elif backbone_type == 'resnet' or backbone_type == 'resnet_beta':
        attention_type = None
        recompute_grad = conv_use_recompute_grad or recompute_grad
        filters_list = [filters,
                        filters,
                        filters * 4]
      elif backbone_type == 'wider_resnet':
        if original_resnet_input_stride * original_resnet_stride < 32:
          # Wider-ResNet uses conv basic blocks except the last stage.
          attention_type = None
          recompute_grad = conv_use_recompute_grad or recompute_grad
          filters_list = [filters * 4,
                          filters * 4]
        else:
          # Wider-ResNet uses an expanded bottleneck block in the last stage.
          attention_type = None
          recompute_grad = conv_use_recompute_grad or recompute_grad
          filters_list = [filters,
                          filters * 2,
                          filters * 4]
      else:
        raise ValueError(backbone_type + ' is not supported.')

      self._drop_path_keep_prob.append(current_drop_path_keep_prob)
      # Apply the residual block.
      block_fn = None
      if use_axial_block:
        # The inputs to block_fn should be activated features.
        block_fn = axial_blocks.AxialBlock(
            filters_list,
            kernel_size=3,
            strides=current_strides,
            atrous_rate=atrous_rate,
            use_squeeze_and_excite=use_squeeze_and_excite,
            use_sac=use_sac,
            bn_layer=bn_layer,
            activation=activation,
            name=current_name[1:],
            conv_kernel_weight_decay=conv_kernel_weight_decay,
            basic_block_second_conv_atrous_rate=(
                basic_block_second_conv_atrous_rate),
            attention_type=attention_type,
            axial_layer_config=axial_layer_config)
      self._recompute_grad.append(recompute_grad)
      utils.safe_setattr(self, current_name, block_fn)

      # Modify the original_resnet_stride according to the strides.
      if index == 0 and original_resnet_stride > 1:
        original_resnet_current_stride *= original_resnet_stride
        # Add absolute positional encoding if we will apply global attention
        # beyond this stride.
        if original_resnet_current_stride == use_global_beyond_stride > 0:
          self._add_absolute_positional_encoding = (
              positional_encodings.AddAbsolutePositionalEncoding(
                  'add_absolute_positional_encoding',
                  positional_encoding_type, bn_layer, conv_kernel_weight_decay))
      if original_resnet_current_stride >= use_transformer_beyond_stride > 0:
        # Apply a dual-path transformer.
        transformer_block_fn = dual_path_transformer.DualPathTransformerLayer(
            name=transformer_current_name[1:],
            filters=int(128 * transformer_expansion),
            activation=activation,
            bn_layer=bn_layer,
            conv_kernel_weight_decay=conv_kernel_weight_decay,
            auxiliary_predictor_func=auxiliary_predictor_func,
            **dual_path_transformer_layer_config)
        utils.safe_setattr(self, transformer_current_name, transformer_block_fn)
      else:
        utils.safe_setattr(self, transformer_current_name, None)
    # Avoid using recompute_grad for the first call that builds the sub-layers.
    # Otherwise, recompute_grad will not track newly built model parameters.
    self._first_building_call = True

  def call(self, inputs, training=False):
    """Performs a forward pass.

    Args:
      inputs: A list of tensors or tuples. The first tensor is a
        pixel_space_input with shape [batch, height, width, pixel_channels].
        The second tensor is memory_space_input with shape [batch, length,
        memory_channels]. The third one is an optional auxiliary_outputs which
        is a tuple containing auxiliary outputs, where each element has the
        dictionary type.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      output: An output [batch, height, width, filters * 4] tensor.
      activated_output: An activated output [batch, height, width, filters * 4]
        tensor.
      memory_space_output: A memory space output [batch, length,
        memory_channels] tensor.
      auxiliary_outputs: An optional tuple containing auxiliary outputs, where
        each element has the dictionary type.

    Raises:
      ValueError: If the length of inputs is not 2 or 3.
    """
    # The pixel space inputs are non-activated features.
    if len(inputs) == 2:
      features, memory_space_output = inputs
      auxiliary_outputs = ()
      return_auxiliary_outputs = False
    elif len(inputs) == 3:
      features, memory_space_output, auxiliary_outputs = inputs
      return_auxiliary_outputs = True
    else:
      raise ValueError('The length of inputs should be either 2 or 3!')

    # Recompute_grad takes only float tensors as inputs. It does not allow
    # bools or boolean tensors. For this reason, we cast training to a float
    # tensor and cast it back after we go through the recompute_grad wrap.
    float_tensor_training = tf.cast(training, tf.float32)

    for index in range(self._num_blocks):
      current_name, transformer_current_name = _get_current_names(index)
      block_fn_no_recompute = getattr(
          self, current_name)
      transformer_block_fn_no_recompute = getattr(
          self, transformer_current_name)
      current_drop_path_keep_prob = self._drop_path_keep_prob[index]

      if self._use_axial_block:
        # Wrap the layer if we want to recompute it in the backward pass.
        if (self._recompute_grad[index] and training):
          # The seed is not actually used since we do not have any random
          # operation in the recomputed function. The purpose of the provided
          # seed is to prevent recompute_grad from generating a new seed
          # variable which is not compatible with model exporting.
          block_fn = recompute_grad_lib.recompute_grad(
              block_fn_no_recompute, seed=tf.constant(0, tf.int32))
        else:
          block_fn = block_fn_no_recompute

        # The inputs to block_fn should be activated features.
        block_fn_inputs = [features, float_tensor_training]
        # We have to define drop_path_masks outside the layer call and pass it
        # into the layer, because tf.recompute_grad (gradient checkpointing)
        # does not allow any randomness within the function call. In addition,
        # recompute_grad functions can only take Tensors as inputs, so we do not
        # pass the drop_path_random_mask (when it is None) into block_fn.
        if current_drop_path_keep_prob < 1.0 and training:
          drop_path_random_mask = drop_path.generate_drop_path_random_mask(
              features, current_drop_path_keep_prob)

          block_fn_inputs.append(drop_path_random_mask)

        # Build the sub-layers when the block_fn is called for the first time.
        # Otherwise, recompute_grad will not track newly built model parameters.
        if self._first_building_call:
          _ = block_fn_no_recompute(tuple(block_fn_inputs))
        # Apply the residual block.
        features = block_fn(tuple(block_fn_inputs))

      if index == 0 and self._add_absolute_positional_encoding is not None:
        features = self._add_absolute_positional_encoding(features,
                                                          training=training)

      if transformer_block_fn_no_recompute is not None:
        # Wrap the layer if we want to recompute it in the backward pass.
        if (self._transformer_use_recompute_grad and training):
          # The seed is not actually used since we do not have any random
          # operation in the recomputed function. The purpose of the provided
          # seed is to prevent recompute_grad from generating a new seed
          # variable which is not compatible with model exporting.
          transformer_block_fn = recompute_grad_lib.recompute_grad(
              transformer_block_fn_no_recompute, seed=tf.constant(0, tf.int32))
        else:
          transformer_block_fn = transformer_block_fn_no_recompute

        transformer_block_fn_input_list = [
            features, memory_space_output, auxiliary_outputs,
            float_tensor_training]
        # We have to define drop_path_masks outside the layer call and pass it
        # into the layer, because recompute_grad (gradient checkpointing) does
        # not allow any randomness within the function call. In addition,
        # recompute_grad functions can only take Tensors as inputs, so we do not
        # pass the drop_path_masks (when they are None) into
        # transformer_block_fn.
        if current_drop_path_keep_prob < 1.0 and training:
          # Drop path random mask for pixel space attention.
          pixel_space_drop_path_mask = drop_path.generate_drop_path_random_mask(
              memory_space_output, current_drop_path_keep_prob)
          # Drop path random mask for memory space attention.
          memory_space_attention_drop_path_mask = (
              drop_path.generate_drop_path_random_mask(
                  memory_space_output, current_drop_path_keep_prob))
          # Drop path random mask for memory space kmeans cross-attention.
          memory_kmeans_attention_drop_path_mask = (
              drop_path.generate_drop_path_random_mask(
                  memory_space_output, current_drop_path_keep_prob))
          # Drop path random mask for memory space feed-forward network.
          memory_space_feed_forward_network_drop_path_mask = (
              drop_path.generate_drop_path_random_mask(
                  memory_space_output, current_drop_path_keep_prob))
          transformer_block_fn_input_list += [
              pixel_space_drop_path_mask,
              memory_space_attention_drop_path_mask,
              memory_kmeans_attention_drop_path_mask,
              memory_space_feed_forward_network_drop_path_mask]

        # Build the sub-layers when the transformer_block_fn is called for the
        # first time. Otherwise, recompute_grad will not track newly built model
        # parameters.
        if self._first_building_call:
          _ = transformer_block_fn_no_recompute(
              tuple(transformer_block_fn_input_list))
        # Apply a dual-path transformer.
        features, memory_space_output, auxiliary_outputs = (
            transformer_block_fn(tuple(transformer_block_fn_input_list)))

    # Now the first call has finished and the sub-layers have been built.
    self._first_building_call = False
    # We also return the non-activated output so that the function is compatible
    # with a decoder that takes a non-activated tensor as input.
    if return_auxiliary_outputs:
      return features, memory_space_output, auxiliary_outputs
    else:
      return features, memory_space_output
