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

"""Implements transformer layers for MaX-DeepLab [1] and kMaX-DeepLab [2].

Dual-path transformer introduces a global memory path in addition to a CNN path,
allowing bi-directional communication with any CNN layers.

k-means cross-attention adopts a cluster-wise argmax instead of spatial-wise
softmax, which aligns to k-means clustering algorithm.

[1] MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
    CVPR 2021.
      Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

[2] k-means Mask Transformer, ECCV 2022.
      Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
      Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
"""

import tensorflow as tf

from deeplab2 import common
from deeplab2.model import utils
from deeplab2.model.layers import activations
from deeplab2.model.layers import convolutions


class AttentionOperation(tf.keras.layers.Layer):
  """Computes standard 1D multi-head attention with query, key, and value."""

  def __init__(self,
               name,
               activation,
               transformer_activation,
               bn_layer=tf.keras.layers.BatchNormalization):
    """Initializes an AttentionOperation layer.

    Args:
      name: A string, the name of this layer.
      activation: A string, type of activation function to apply.
      transformer_activation: A string, type of activation function for
        self-attention. Support 'sigmoid' and 'softmax'.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
    """
    super(AttentionOperation, self).__init__(name=name)
    # batch_norm_similarity has shape [batch, num_heads, num_query, num_key],
    # where num_query and num_key usually equals to height or width or length,
    # i.e., spatial dimensions, so batch norm is applied to axis=1 only.
    self._batch_norm_similarity = bn_layer(axis=1, name='batch_norm_similarity')
    # batch_norm_retrieved_value is done on shape [batch, num_heads, length,
    # value_channels], which will be reshaped to the output shape [batch,
    # length, value_channels * num_heads], so we apply batch norm on the
    # effective channel dimension -- value_channels * num_heads.
    self._batch_norm_retrieved_value = bn_layer(
        axis=[1, 3], name='batch_norm_retrieved_value')
    self._activation_fn = activations.get_activation(activation)
    self._transformer_activation_fn = activations.get_activation(
        transformer_activation)

  def call(self, inputs, training=False):
    """Performs an AttentionOperation.

    Args:
      inputs: A tuple of (query, key, value), where query is [batch, num_head,
        query_length, channels] tensor, key is a [batch, num_head, key_length,
        channels] tensor, and value is a [batch, key_length, num_head,
        value_channels] tensor.
      training: A boolean, whether the model is in training mode.

    Returns:
      output: A [batch, query_length, num_head * value_channels] tensor, the
        retrieved value.
    """
    # Decode query, key, and value from inputs.
    query, key, value = inputs
    # Compute attention similarity.
    similarity_logits = tf.einsum('bhld,bhmd->bhlm', query, key)
    similarity_logits = self._batch_norm_similarity(
        similarity_logits, training=training)
    # Apply a transformer attention activation function, e.g. softmax.
    attention_weights = self._transformer_activation_fn(similarity_logits)
    # Retrieve the value content.
    retrieved_value = tf.einsum(
        'bhlm,bmhd->bhld', attention_weights, value)
    retrieved_value = self._batch_norm_retrieved_value(
        retrieved_value, training=training)
    retrieved_value = self._activation_fn(retrieved_value)
    # Reshape the output.
    return utils.transpose_and_reshape_for_attention_operation(
        retrieved_value)


class DualPathTransformerLayer(tf.keras.layers.Layer):
  """Applies a transformer layer, as proposed in MaX-DeepLab models [1,2].

  Dual-path transformer layer takes a pixel space input and a memory space
  input, and performs memory2pixel attention, pixel2memory attention, and
  memory2memory self-attention. Note that the pixel2pixel self-attention or
  convolution in the pixel space is implemented in axial_layers.py and
  axial_blocks.py. Thus, the pixel2pixel operation is not included in this
  DualPathTransformerLayer implementation. Please use this class together with
  a residual block with axial-attention, global-attention, or convolution in
  order to construct the full dual path transformer in the paper.

  The flag "use_kmeans_cross_attention" enables k-means cross-attention, which
  regards the memory (object query) as cluster center, and updates them in a
  k-means clustering manner.

  [1] MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
      CVPR 2021.
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

  [2] k-means Mask Transformer, ECCV 2022.
      Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
      Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  def __init__(self,
               name='dual_path_transformer_layer',
               activation='relu',
               filters=128,
               num_heads=8,
               bottleneck_expansion=2,
               key_expansion=1,
               value_expansion=2,
               feed_forward_network_channels=2048,
               use_memory_self_attention=True,
               use_memory2pixel_feedback_attention=True,
               use_pixel2memory_feedback_attention=True,
               use_kmeans_cross_attention=False,
               transformer_activation='softmax',
               bn_layer=tf.keras.layers.BatchNormalization,
               conv_kernel_weight_decay=0.0,
               auxiliary_predictor_func=None):
    """Initializes a DualPathTransformerLayer.

    This function implements a dual path transformer layer between a pixel space
    and a memory space, as described in the MaX-DeepLab paper. In this dual path
    transformer, the memory2pixel cross attention and the memory self-attention
    share a single activation, e.g. softmax.

    The flag "use_kmeans_cross_attention" enables k-means cross-attention
    proposed in the kMaX-DeepLab paper, which regards the memory (object query)
    as cluster center, and updates them in a k-means clustering manner.

    Reference:
      MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
        CVPR 2021. https://arxiv.org/abs/2012.00759
          Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

      k-means Mask Transformer, ECCV 2022.
        Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
        Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

    Args:
      name: A string, the name of this dual path transformer layer.
      activation: A string, type of activation function to apply.
      filters: An integer, the base number of channels for the layer.
      num_heads: An integer, the number of heads in multi-head attention.
      bottleneck_expansion: A float, the channel expansion ratio for the
        bottleneck.
      key_expansion: A float, the channel expansion ratio for keys.
      value_expansion: A float, the channel expansion ratio for values.
      feed_forward_network_channels: An integer, the number of channels for the
        feed_forward_network. Zero means no feed_forward_network will be
        applied.
      use_memory_self_attention: A boolean, whether to apply the memory space
        self-attention.
      use_memory2pixel_feedback_attention: A boolean, whether to apply the
        memory2pixel feedback attention.
      use_pixel2memory_feedback_attention: A boolean, whether to apply the
        pixel2memory feedback attention.
      use_kmeans_cross_attention: A boolean, whether to apply the kmeans
        cross-attention.
      transformer_activation: A string, type of activation function for
        self-attention. Support 'sigmoid' and 'softmax'.
      bn_layer: A tf.keras.layers.Layer that computes the normalization
        (default: tf.keras.layers.BatchNormalization).
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.
      auxiliary_predictor_func: A callable function that returns an
        initialization of auxiliary predictor.

    Raises:
      ValueError: If filters * key_expansion is not divisible by num_heads.
      ValueError: If filters * value_expansion is not divisible by num_heads.
      ValueError: If both use_memory2pixel_feedback_attention and
        use_kmeans_cross_attention are False.
      ValueError: If use_kmeans_cross_attention is True but
        auxiliary_predictor_func is None.
    """
    super(DualPathTransformerLayer, self).__init__(name=name)

    bottleneck_channels = int(round(filters * bottleneck_expansion))
    total_key_depth = int(round(filters * key_expansion))
    total_value_depth = int(round(filters * value_expansion))

    if total_key_depth % num_heads:
      raise ValueError('Total_key_depth should be divisible by num_heads.')

    if total_value_depth % num_heads:
      raise ValueError('Total_value_depth should be divisible by num_heads.')

    if not (use_memory2pixel_feedback_attention or use_kmeans_cross_attention):
      raise ValueError('At least one of use_memory2pixel_feedback_attention or'
                       ' use_kmeans_cross_attention needs to be enabled.')

    # Compute query key value with one convolution and a batch norm layer. The
    # initialization std is standard transformer initialization (without batch
    # norm), as used in SASA and ViT. In our case, we use batch norm by default,
    # so it does not require careful tuning. If one wants to remove all batch
    # norms in axial attention, this standard initialization should still be
    # good, but a more careful initialization is encouraged.
    initialization_std = bottleneck_channels ** -0.5

    self._memory_conv1_bn_act = convolutions.Conv1D(
        bottleneck_channels, 'memory_conv1_bn_act',
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=activation,
        conv_kernel_weight_decay=conv_kernel_weight_decay)

    self._pixel_conv1_bn_act = convolutions.Conv1D(
        bottleneck_channels, 'pixel_conv1_bn_act',
        use_bias=False,
        use_bn=True,
        bn_layer=bn_layer,
        activation=activation,
        conv_kernel_weight_decay=conv_kernel_weight_decay)

    # We always compute the query for memory space, since it gathers information
    # from the pixel space and thus cannot be removed. We compute the key and
    # value for memory space only when they are necessary (i.e. either
    # use_memory_self_attention or use_pixel2memory_feedback_attention).
    if use_memory_self_attention or use_pixel2memory_feedback_attention:
      self._memory_qkv_conv_bn = convolutions.Conv1D(
          total_key_depth * 2 + total_value_depth, 'memory_qkv_conv_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation='none',
          conv_kernel_weight_decay=conv_kernel_weight_decay,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=initialization_std))
    elif use_memory2pixel_feedback_attention:
      # Compute memory query only if memory key and value are not used.
      self._memory_query_conv_bn = convolutions.Conv1D(
          total_key_depth,
          'memory_query_conv_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation='none',
          conv_kernel_weight_decay=conv_kernel_weight_decay,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=initialization_std))

    # For the pixel space, we always compute the key and value, since they
    # provide information for the memory space and thus cannot be removed. We
    # compute the query for pixel space only when it is necessary (i.e.
    # use_pixel2memory_feedback_attention is True).
    if use_pixel2memory_feedback_attention:
      self._pixel_qkv_conv_bn = convolutions.Conv1D(
          total_key_depth * 2 + total_value_depth, 'pixel_qkv_conv_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation='none',
          conv_kernel_weight_decay=conv_kernel_weight_decay,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=initialization_std))
    elif use_memory2pixel_feedback_attention:
      self._pixel_kv_conv_bn = convolutions.Conv1D(
          total_key_depth + total_value_depth, 'pixel_kv_conv_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation='none',
          conv_kernel_weight_decay=conv_kernel_weight_decay,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=initialization_std))
    else:
      # In this case, only k-means cross attention is enabled, and thus we only
      # need the value of pixel to update memory.
      self._pixel_v_conv_bn = convolutions.Conv1D(
          total_value_depth, 'pixel_v_conv_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=bn_layer,
          activation='none',
          conv_kernel_weight_decay=conv_kernel_weight_decay,
          kernel_initializer=tf.keras.initializers.TruncatedNormal(
              stddev=initialization_std))

    if use_memory2pixel_feedback_attention or use_memory_self_attention:
      self._memory_attention = AttentionOperation(
          'memory_attention', activation, transformer_activation,
          bn_layer=bn_layer)
    if use_pixel2memory_feedback_attention:
      self._pixel_attention = AttentionOperation(
          'pixel_attention', activation, transformer_activation,
          bn_layer=bn_layer)

    if use_kmeans_cross_attention:
      # If kmeans cross-attention is used, we perform an auxiliary prediction.
      if auxiliary_predictor_func is None:
        raise ValueError('auxiliary_predictor_func should not be None when'
                         ' using kmeans cross attention.')
      self._auxiliary_clustering_predictor = auxiliary_predictor_func()

    self._use_memory_self_attention = use_memory_self_attention
    self._use_memory2pixel_feedback_attention = (
        use_memory2pixel_feedback_attention)
    self._use_pixel2memory_feedback_attention = (
        use_pixel2memory_feedback_attention)
    self._use_kmeans_cross_attention = use_kmeans_cross_attention
    self._bottleneck_channels = bottleneck_channels
    self._total_key_depth = total_key_depth
    self._total_value_depth = total_value_depth
    self._num_heads = num_heads
    self._bn_layer = bn_layer
    self._conv_kernel_weight_decay = conv_kernel_weight_decay
    self._activation = activation
    self._activation_fn = activations.get_activation(activation)
    self._feed_forward_network_channels = feed_forward_network_channels

  def build(self, input_shape_list):
    pixel_shape, memory_shape = input_shape_list[:2]
    # Here we follow ResNet bottleneck blocks: we apply a batch norm with gamma
    # initialized at zero, followed by drop path and an activation function.
    # Initializing this gamma at zero ensures that at random initialization of
    # the model, the skip connections dominate all residual blocks. In this way,
    # all the skip connections construct an identity mapping that passes the
    # gradients (without any distortion from the randomly initialized blocks) to
    # all residual blocks. This helps training at early epochs.
    # Reference: "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour".
    # https://arxiv.org/abs/1706.02677
    if (self._use_memory2pixel_feedback_attention or
        self._use_memory_self_attention):
      self._memory_conv3_bn = convolutions.Conv1D(
          memory_shape[-1], 'memory_conv3_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=self._bn_layer,
          bn_gamma_initializer='zeros',
          activation='none',
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)

    if self._feed_forward_network_channels > 0:
      self._memory_ffn_conv1_bn_act = convolutions.Conv1D(
          self._feed_forward_network_channels, 'memory_ffn_conv1_bn_act',
          use_bias=False,
          use_bn=True,
          bn_layer=self._bn_layer,
          activation=self._activation,
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)
      # Again, we follow ResNet bottleneck blocks: we apply a batch norm with
      # gamma initialized at zero, followed by drop path and an activation
      # function.
      self._memory_ffn_conv2_bn = convolutions.Conv1D(
          memory_shape[-1], 'memory_ffn_conv2_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=self._bn_layer,
          bn_gamma_initializer='zeros',
          activation='none',
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)
    if self._use_pixel2memory_feedback_attention:
      self._pixel_conv3_bn = convolutions.Conv1D(
          pixel_shape[-1], 'pixel_conv3_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=self._bn_layer,
          bn_gamma_initializer='zeros',
          activation='none',
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)

    if self._use_kmeans_cross_attention:
      self._kmeans_memory_batch_norm_retrieved_value = self._bn_layer(
          axis=-1, name='kmeans_memory_batch_norm_retrieved_value')
      self._kmeans_memory_conv3_bn = convolutions.Conv1D(
          memory_shape[-1], 'kmeans_memory_conv3_bn',
          use_bias=False,
          use_bn=True,
          bn_layer=self._bn_layer,
          bn_gamma_initializer='zeros',
          activation='none',
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)

  def kmeans_assignment_step(self, pixel_feature, cluster_centers,
                             num_mask_slots, training):
    auxiliary_result_dict = self._auxiliary_clustering_predictor(
        {
            'feature_panoptic': pixel_feature,
            'feature_semantic': pixel_feature,
            'transformer_class_feature': cluster_centers,
            'transformer_mask_feature': cluster_centers
        },
        training=training)
    # A cluster-wise argmax is applied to convert the attention logits to
    # clustering results, which serve as attention weights.
    clustering_result = tf.argmax(
        auxiliary_result_dict[common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY], axis=-1)
    clustering_result = tf.cast(
        tf.one_hot(clustering_result, depth=num_mask_slots, axis=-1),
        tf.float32)
    clustering_result = tf.stop_gradient(clustering_result)
    return auxiliary_result_dict, clustering_result

  def kmeans_update_memory_step(self, clustering_result, pixel_value, training):
    kmeans_memory_space = tf.einsum('blm,bld->bmd', clustering_result,
                                    pixel_value)
    kmeans_memory_space = self._kmeans_memory_batch_norm_retrieved_value(
        kmeans_memory_space, training=training)
    kmeans_memory_space = self._kmeans_memory_conv3_bn(
        kmeans_memory_space, training=training)
    return kmeans_memory_space

  def call(self, inputs):
    """Performs a forward pass.

    We have to define drop_path_masks outside the layer call and pass it into
    the layer call, because recompute_grad (gradient checkpointing) does not
    allow any randomness within the function call. In addition, recompute_grad
    only supports float tensors as inputs. For this reason, the training flag
    should be also passed as a float tensor. For the same reason, we cannot
    support passing drop_path_random_mask as None. Instead, we ask the users to
    pass only the first two tensors when drop path is not used.

    Args:
      inputs: A tuple of 4 or 8 tensors, containing
        pixel_space_input should be a [batch, height, width,
          pixel_space_channels] tensor.
        memory_space_input should be a [batch, num_memory,
          memory_space_channels] tensor.
        auxiliary_outputs should be a tuple containing auxiliary outputs, where
          each element has the dictionary type.
        float_tensor_training should be a float tensor of 0.0 or 1.0, whether
          the model is in training mode.
        (optional) pixel_space_drop_path_mask is a drop path mask tensor of
          shape [batch, 1, 1] for the pixel space.
        (optional) memory_space_attention_drop_path_mask is a drop path mask
          tensor of shape [batch, 1, 1] for the memory space.
        (optional) memory_kmeans_attention_drop_path_mask is a drop path mask
          tensor of shape [batch, 1, 1] for the memory space.
        (optional) memory_space_feed_forward_network_drop_path_mask is a drop
          path mask tensor of shape [batch, 1, 1] for the memory space feed
          forward network.

    Returns:
      pixel_space_output: A [batch, num_pixel, pixel_space_channels] tensor.
      memory_space_output: A [batch, num_memory, memory_space_channels]
        tensor.
      auxiliary_outputs: A tuple containing auxiliary outputs, where each
        element has the dictionary type.

    Raises:
      ValueError: If the length of inputs is not 4 or 8.
    """
    if len(inputs) not in (4, 8):
      raise ValueError('The length of inputs should be either 4 or 8.')

    # Unpack the inputs.
    (pixel_space_input, memory_space_input, auxiliary_outputs,
     float_tensor_training,
     pixel_space_drop_path_mask, memory_space_attention_drop_path_mask,
     memory_kmeans_attention_drop_path_mask,
     memory_space_feed_forward_network_drop_path_mask) = (
         utils.pad_sequence_with_none(inputs, target_length=8))

    # Recompute_grad takes only float tensors as inputs. It does not allow
    # bools or boolean tensors. For this reason, we cast training to a float
    # tensor outside this call, and now we cast it back to a boolean tensor.
    training = tf.cast(float_tensor_training, tf.bool)

    # Decode the inputs shapes.
    pixel_shape = pixel_space_input.get_shape().as_list()
    memory_shape = memory_space_input.get_shape().as_list()

    # Flatten the pixel_space_input.
    pixel_space_input = tf.reshape(
        pixel_space_input,
        [-1, pixel_shape[1] * pixel_shape[2], pixel_shape[3]])

    # Similar to the ResNet bottleneck design, we do an input down projection
    # in both the pixel space and the memory space.
    memory_space = self._memory_conv1_bn_act(memory_space_input,
                                             training=training)

    # Pixel space input is not activated.
    pixel_space = self._pixel_conv1_bn_act(
        self._activation_fn(pixel_space_input), training=training)

    if (self._use_memory_self_attention or
        self._use_pixel2memory_feedback_attention):
      memory_space_qkv = self._memory_qkv_conv_bn(memory_space,
                                                  training=training)
      # Split, reshape, and transpose the query, key, and value.
      memory_query, memory_key, memory_value = (
          tf.split(memory_space_qkv, [
              self._total_key_depth, self._total_key_depth,
              self._total_value_depth], axis=-1))
      memory_key = utils.reshape_and_transpose_for_attention_operation(
          memory_key, self._num_heads)
      memory_value = tf.reshape(memory_value, [
          -1, memory_shape[1], self._num_heads,
          self._total_value_depth // self._num_heads])
      # Reshape and transpose the query.
      memory_query = utils.reshape_and_transpose_for_attention_operation(
          memory_query, self._num_heads)
    elif self._use_memory2pixel_feedback_attention:
      # Compute memory query only if memory key and value are not used.
      memory_query = self._memory_query_conv_bn(memory_space, training=training)
      # Reshape and transpose the query.
      memory_query = utils.reshape_and_transpose_for_attention_operation(
          memory_query, self._num_heads)

    if self._use_pixel2memory_feedback_attention:
      pixel_space_qkv = self._pixel_qkv_conv_bn(pixel_space,
                                                training=training)
      # Split the query, key, and value.
      pixel_query, pixel_key, pixel_value = tf.split(
          pixel_space_qkv, [
              self._total_key_depth, self._total_key_depth,
              self._total_value_depth], axis=-1)
      pixel_query = utils.reshape_and_transpose_for_attention_operation(
          pixel_query, self._num_heads)
      # Reshape and transpose the key and the value.
      pixel_key = utils.reshape_and_transpose_for_attention_operation(
          pixel_key, self._num_heads)
    elif self._use_memory2pixel_feedback_attention:
      pixel_space_kv = self._pixel_kv_conv_bn(pixel_space, training=training)
      # Split the key and the value.
      pixel_key, pixel_value = tf.split(pixel_space_kv, [
          self._total_key_depth, self._total_value_depth], axis=-1)
      # Reshape and transpose the key and the value.
      pixel_key = utils.reshape_and_transpose_for_attention_operation(
          pixel_key, self._num_heads)
    else:
      # In this case, only k-means cross attention is enabled, and thus we only
      # need the value of pixel to update memory.
      pixel_value = self._pixel_v_conv_bn(pixel_space, training=training)
    pixel_value = tf.reshape(pixel_value, [
        -1, pixel_shape[1] * pixel_shape[2], self._num_heads,
        self._total_value_depth // self._num_heads
    ])

    memory_space_output = memory_space_input
    pixel_space_output = pixel_space_input

    # Perform kmeans cross-attention.
    if self._use_kmeans_cross_attention:
      # Assignment step.
      pixel_space2d = tf.reshape(
          pixel_space,
          [-1, pixel_shape[1], pixel_shape[2], self._bottleneck_channels])
      auxiliary_result_dict, clustering_result = self.kmeans_assignment_step(
          pixel_space2d, memory_space, memory_shape[1], training)
      clustering_result = tf.reshape(clustering_result,
                                     [-1, pixel_shape[1] * pixel_shape[2],
                                      memory_shape[1]])
      # Add to auxiliary_outputs.
      auxiliary_outputs = auxiliary_outputs + (auxiliary_result_dict,)
      # Update step.
      pixel_value_single_head = tf.reshape(
          pixel_value,
          [-1, pixel_shape[1] * pixel_shape[2], self._total_value_depth])
      kmeans_memory_space = self.kmeans_update_memory_step(
          clustering_result, pixel_value_single_head, training)
      if memory_kmeans_attention_drop_path_mask is not None:
        kmeans_memory_space = (
            kmeans_memory_space * memory_kmeans_attention_drop_path_mask)
      memory_space_output = memory_space_output + kmeans_memory_space

    # Perform M2P/M2M attention.
    # Compute memory space attention.
    memory_attention_key = []
    memory_attention_value = []
    if self._use_memory2pixel_feedback_attention:
      # If memory self attention is not used, then only memory2pixel cross
      # attention is used for the memory space. In this case, the key and the
      # value are simply pixel_key and pixel_value.
      memory_attention_key.append(pixel_key)
      memory_attention_value.append(pixel_value)
    if self._use_memory_self_attention:
      memory_attention_key.append(memory_key)
      memory_attention_value.append(memory_value)

    if (self._use_memory2pixel_feedback_attention or
        self._use_memory_self_attention):
      # If we also use memory self attention, the key and the value are the
      # concatenation of keys and values in both the pixel space and the
      # memory space.
      memory_attention_key = tf.concat(memory_attention_key, axis=2)
      memory_attention_value = tf.concat(memory_attention_value, axis=1)
      memory_space = self._memory_attention(
          (memory_query, memory_attention_key, memory_attention_value),
          training=training)
      memory_space = self._memory_conv3_bn(memory_space, training=training)

      if memory_space_attention_drop_path_mask is not None:
        memory_space = memory_space * memory_space_attention_drop_path_mask
      memory_space_output = memory_space_output + memory_space

    memory_space_output = self._activation_fn(memory_space_output)

    # Apply an optional feed-forward network to the memory space.
    if self._feed_forward_network_channels > 0:
      memory_space = self._memory_ffn_conv1_bn_act(memory_space_output,
                                                   training=training)
      memory_space = self._memory_ffn_conv2_bn(memory_space,
                                               training=training)
      if memory_space_feed_forward_network_drop_path_mask is not None:
        memory_space = (memory_space *
                        memory_space_feed_forward_network_drop_path_mask)
      memory_space_output = self._activation_fn(
          memory_space_output + memory_space)

    # Perform P2M attention.
    # Compute pixel space attention and the output projection only when
    # pixel2memory_feedback_attention is used.
    if self._use_pixel2memory_feedback_attention:
      pixel_space = self._pixel_attention(
          (pixel_query, memory_key, memory_value), training=training)
      pixel_space = self._pixel_conv3_bn(pixel_space, training=training)
      if pixel_space_drop_path_mask is not None:
        pixel_space = pixel_space * pixel_space_drop_path_mask
      pixel_space_output = pixel_space_input + pixel_space

    # Reshape back to 4D.
    pixel_space_output = tf.reshape(
        pixel_space_output,
        [-1, pixel_shape[1], pixel_shape[2], pixel_shape[3]])

    # Return the pixel space output, memory space output, and auxiliary outputs.
    return pixel_space_output, memory_space_output, auxiliary_outputs
