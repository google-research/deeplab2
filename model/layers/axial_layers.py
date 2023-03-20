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

"""Implements Axial-Attention layers proposed in Axial-DeepLab.

Axial-Attention factorizes 2D self-attention into two 1D self-attentions, so
that it can be applied on large inputs. Axial-Attention is typically used to
replace 3x3 convolutions in a bottleneck residual block.

[1] Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
    ECCV 2020 Spotlight.
      Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
      Liang-Chieh Chen.
"""

import numpy as np
import tensorflow as tf

from deeplab2.model import utils
from deeplab2.model.layers import activations
from deeplab2.model.layers import positional_encodings


class AxialAttention(tf.keras.layers.Layer):
  """An axial-attention layer."""

  def __init__(self,
               query_shape=129,
               memory_flange=32,
               total_key_depth=512,
               total_value_depth=1024,
               num_heads=8,
               name='axial_attention',
               use_query_rpe_similarity=True,
               use_key_rpe_similarity=True,
               use_content_similarity=True,
               retrieve_value_rpe=True,
               retrieve_value_content=True,
               initialization_std_for_query_key_rpe=1.0,
               initialization_std_for_value_rpe=1.0,
               self_attention_activation='softmax',
               bn_layer=tf.keras.layers.BatchNormalization,
               conv_kernel_weight_decay=0.0):
    """Initializes an axial-attention layer.

    This function is designed to support both global and local axial-attention
    in a unified way. If query_shape is larger than the length of input, a
    global attention is applied. If query_shape is smaller than the length of
    input, a local attention is applied. In this case, the input is divided into
    blocks of length query_shape, padded by memory_flange on both sides. Then,
    local attention is applied within each query block. The choice of
    query_shape does not affect the output value but affects computation
    efficiency and memory usage. In general, use global attention (large
    query_shape) if possible. Local axial-attention has not been supported yet.

    Args:
      query_shape: An integer, the block size for local axial attention.
        Defaults to 129 since 129 is usually the largest feature map where we do
        global attention (1025 with stride 8, or 2049 with stride 16).
      memory_flange: An integer, the memory flange padded to each query block in
        local attention. It has no effect in global attention. Defaults to 32,
        which is equivalent to a span of 65 in Aixal-DeepLab paper -- A pixel
        can see 32 pixels on its left and 32 pixels on its right.
      total_key_depth: An integer, the total depth of keys, which is also the
        depth of queries and the depth of key (query) positional encodings.
      total_value_depth: An integer, the total depth of the values, which is
        also the depth of value positional encodings.
      num_heads: An integer, the number of heads in multi-head attention.
      name: A string, the name of this axial attention layer.
      use_query_rpe_similarity: A boolean, whether to use the attention
        similarity between the queries and the relative positional encodings.
      use_key_rpe_similarity: A boolean, whether to use the attention similarity
        between the keys and the relative positional encodings.
      use_content_similarity: A boolean, whether to use the content similarity
        between the queries and the keys.
      retrieve_value_rpe: A boolean, whether to retrieve the relative positional
        encodings of the values.
      retrieve_value_content: A boolean, whether to retrieve the content of the
        values.
      initialization_std_for_query_key_rpe: A float, the initialization std for
        the relative positional encodings of the queries and keys.
      initialization_std_for_value_rpe: A float, the initialization std for the
        relative positional encodings of the values.
      self_attention_activation: A string, type of activation function for
        self-attention. Support 'sigmoid' and 'softmax'.
      bn_layer: A tf.keras.layers.Layer that computes the normalization
        (default: tf.keras.layers.BatchNormalization).
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.

    Returns:
      output: A [batch, length, total_value_depth] tensor.

    Raises:
      ValueError: If none of the three similarities (use_query_rpe_similarity,
        use_key_rpe_similarity, use_content_similarity) is used.
      ValueError: If neither of value content or value rpe is retrieved.
      ValueError: If self_attention_activation is not supported.
      ValueError: If total_key_depth is not divisible by num_heads.
      ValueError: If total_value_depth is not divisible by num_heads.
    """
    # Validate the attention similarity choices.
    if not any([
        use_content_similarity, use_key_rpe_similarity, use_query_rpe_similarity
    ]):
      raise ValueError(
          'Should use at least one similarity to compute attention.')

    # Validate the retrieve value choices.
    if not retrieve_value_content and not retrieve_value_rpe:
      raise ValueError('Should retrieve at least one of content or rpe.')

    if total_key_depth % num_heads:
      raise ValueError('Total_key_depth should be divisible by num_heads.')

    if total_value_depth % num_heads:
      raise ValueError('Total_value_depth should be divisible by num_heads.')

    super(AxialAttention, self).__init__(name=name)
    self._query_shape = query_shape
    self._memory_flange = memory_flange
    self._total_key_depth = total_key_depth
    self._total_value_depth = total_value_depth
    self._num_heads = num_heads
    self._use_query_rpe_similarity = use_query_rpe_similarity
    self._use_key_rpe_similarity = use_key_rpe_similarity
    self._use_content_similarity = use_content_similarity
    self._retrieve_value_rpe = retrieve_value_rpe
    self._retrieve_value_content = retrieve_value_content
    self._initialization_std_for_query_key_rpe = (
        initialization_std_for_query_key_rpe)
    self._initialization_std_for_value_rpe = initialization_std_for_value_rpe
    self._self_attention_activation = self_attention_activation
    self._conv_kernel_weight_decay = conv_kernel_weight_decay

    self._batch_norm_qkv = bn_layer(axis=-1, name='batch_norm_qkv')
    self._batch_norm_similarity = bn_layer(
        axis=[0, 2], name='batch_norm_similarity')
    self._batch_norm_retrieved_output = bn_layer(
        axis=[0, 2, 4], name='batch_norm_retrieved_output')

    self._key_depth_per_head = total_key_depth // num_heads
    self._attention_activate_fn = activations.get_activation(
        self_attention_activation)

  def build(self, input_shape):
    """Builds axial-attention layer weights.

    Args:
      input_shape: An integer list of length 3, the shape of the input tensor.

    Raises:
      NotImplementedError: Local axial-attention has not been implemented. It is
        triggered if query_shape is less than input_shape.
    """

    # Apply global attention if query_shape is larger than input_shape[1].
    if self._query_shape >= input_shape[1]:
      self._query_shape = input_shape[1]
      self._memory_flange = 0
    else:
      raise NotImplementedError('Local axial attention has not been '
                                'implemented yet.')
    self._memory_shape = self._query_shape + 2 * self._memory_flange

    # Compute query key value with one convolution and an optional batch norm.
    # The initialization std is standard transformer initialization (without
    # batch norm), as used in SASA and ViT. In our case, we use batch norm by
    # default, so it does not require careful tuning. If one wants to remove
    # all batch norms in axial attention, this standard initialization should
    # still be good, but a more careful initialization is encouraged.
    self.qkv_kernel = self.add_weight(
        name='qkv_kernel',
        shape=[input_shape[-1],
               self._total_key_depth * 2 + self._total_value_depth],
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=input_shape[-1]**-0.5),
        regularizer=tf.keras.regularizers.l2(self._conv_kernel_weight_decay))

    if self._use_query_rpe_similarity:
      self._query_rpe = positional_encodings.RelativePositionalEncoding(
          self._query_shape,
          self._memory_shape,
          self._key_depth_per_head,
          'query_rpe',
          initialization_std=self._initialization_std_for_query_key_rpe,
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)

    if self._use_key_rpe_similarity:
      self._key_rpe = positional_encodings.RelativePositionalEncoding(
          self._query_shape,
          self._memory_shape,
          self._key_depth_per_head,
          'key_rpe',
          initialization_std=self._initialization_std_for_query_key_rpe,
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)

    if self._retrieve_value_rpe:
      self._value_rpe = positional_encodings.RelativePositionalEncoding(
          self._query_shape,
          self._memory_shape,
          self._total_value_depth // self._num_heads,
          'value_rpe',
          initialization_std=self._initialization_std_for_value_rpe,
          conv_kernel_weight_decay=self._conv_kernel_weight_decay)

  def call(self, input_tensor, training=False):
    """Performs a forward pass.

    Args:
      input_tensor: An input [batch, length, channel] tensor.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      output: An output [batch, length, total_value_depth] tensor.
    """
    # Alternatively, the einsum can be implemented as a 1x1 convolution.
    # However, it is not obvious which implementation is more efficient (without
    # careful benchmarking), so we use einsum for its flexibility and
    # consistency with other parts of the function.
    query_key_value = tf.einsum(
        'nlc,cd->nld', input_tensor, self.qkv_kernel, name='compute_qkv')
    query_key_value = self._batch_norm_qkv(query_key_value, training=training)

    # Split query key value.
    query, key, value = tf.split(
        query_key_value,
        [self._total_key_depth, self._total_key_depth, self._total_value_depth],
        axis=-1)

    # Reshape the query, key, and value.
    query = tf.reshape(query, [-1, self._query_shape, self._num_heads,
                               self._key_depth_per_head])
    query = tf.transpose(a=query, perm=[0, 2, 1, 3])
    key = tf.reshape(key, [-1, np.prod(self._memory_shape), self._num_heads,
                           self._key_depth_per_head])
    key = tf.transpose(a=key, perm=[0, 2, 1, 3])
    value = tf.reshape(value, [-1, np.prod(self._memory_shape), self._num_heads,
                               self._total_value_depth // self._num_heads])

    # Gather all similarity logits into a list.
    similarity_logits = []

    # Compute the content similarity term: q * k.
    if self._use_content_similarity:
      content_similarity = tf.einsum(
          'bhld,bhmd->bhlm', query, key, name='content_similarity')
      similarity_logits.append(content_similarity)

    # Compute the query rpe similarity term: q * rpe.
    if self._use_query_rpe_similarity:
      query_rpe = self._query_rpe(None)
      query_rpe_similarity = tf.einsum(
          'bhld,lmd->bhlm', query, query_rpe, name='query_rpe_similarity')
      similarity_logits.append(query_rpe_similarity)

    # Compute the key rpe similarity term: k * rpe.
    if self._use_key_rpe_similarity:
      key_rpe = self._key_rpe(None)
      key_rpe_similarity = tf.einsum(
          'bhmd,lmd->bhlm', key, key_rpe, name='key_rpe_similarity')
      similarity_logits.append(key_rpe_similarity)

    # Apply an optional batch norm to the similarities and sum them.
    similarity_logits = tf.stack(similarity_logits)
    similarity_logits = self._batch_norm_similarity(similarity_logits,
                                                    training=training)
    similarity_logits = tf.reduce_sum(input_tensor=similarity_logits, axis=0)

    # Apply an attention activation function, e.g. softmax.
    weights = self._attention_activate_fn(similarity_logits)

    # Gather retrieved values or rpes into a list.
    retrieve_list = []

    # Retrieve the content of the attended value.
    if self._retrieve_value_content:
      retrieved_content = tf.einsum(
          'bhlm,bmhd->bhld', weights, value, name='retrieve_value_content')
      retrieve_list.append(retrieved_content)

    # Retrieve the relative position of the attended value.
    if self._retrieve_value_rpe:
      value_rpe = self._value_rpe(None)
      retrieved_rpe = tf.einsum(
          'bhlm,lmd->bhld', weights, value_rpe, name='retrieve_value_rpe')
      retrieve_list.append(retrieved_rpe)

    # Apply batch norms to retrieved contents and rpes respectively.
    retrieved_output = tf.stack(retrieve_list)
    retrieved_output = self._batch_norm_retrieved_output(retrieved_output,
                                                         training=training)
    # Additive contents and rpes.
    retrieved_output = tf.reduce_sum(input_tensor=retrieved_output, axis=0)

    # Combine the heads by transposing and reshaping the tensor.
    retrieved_output = utils.transpose_and_reshape_for_attention_operation(
        retrieved_output)

    return retrieved_output


class AxialAttention2D(tf.keras.layers.Layer):
  """Sequentially applies height-axis and width-axis axial-attention."""

  def __init__(self,
               strides=1,
               filters=512,
               name='attention',
               key_expansion=1,
               value_expansion=2,
               query_shape=(129, 129),
               memory_flange=(32, 32),
               **kwargs):
    """Initializes an AxialAttention2D layer.

    Args:
      strides: An integer, the stride for the output, usually 1 or 2.
      filters: An integer, the base number of channels for the layer.
      name: A string, the name of the attention layer.
      key_expansion: A float, the channel expansion ratio for keys.
      value_expansion: A float, the channel expansion ratio for values.
      query_shape: An integer, the maximum query shape for both the height axis
        and the width axis.
      memory_flange: An integer list of length 2. The memory flange for the
        height axis and the width axis.
      **kwargs: A dictionary of keyword arguments passed to height-axis,
        width-axis, and 2D global AxialAttention.

    Returns:
      output: A [batch, strided height, strided width, output_channels] tensor.
    """
    super(AxialAttention2D, self).__init__(name=name)
    total_key_depth = int(round(filters * key_expansion))
    total_value_depth = int(round(filters * value_expansion))
    self._strides = strides
    self._total_key_depth = total_key_depth
    self._total_value_depth = total_value_depth
    self._height_axis = AxialAttention(
        total_key_depth=total_key_depth,
        total_value_depth=total_value_depth,
        query_shape=query_shape[0],
        memory_flange=memory_flange[0],
        name='height_axis',
        **kwargs)
    self._width_axis = AxialAttention(
        total_key_depth=total_key_depth,
        total_value_depth=total_value_depth,
        query_shape=query_shape[1],
        memory_flange=memory_flange[1],
        name='width_axis',
        **kwargs)

  def call(self, inputs, training=False):
    """Performs a forward pass.

    Args:
      inputs: An input [batch, height, width, channel] tensor.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      output: An output [batch, strided_height, strided_width,
        filters * value_expansion] tensor.
    """
    _, height, width, channel = inputs.get_shape().as_list()

    # Transpose and reshape the width axis to the batch dimension.
    x = tf.transpose(a=inputs, perm=[0, 2, 1, 3])
    x = tf.reshape(x, [-1, height, channel])
    x = self._height_axis(x, training=training)
    # Reshape and transpose back to a 4D tensor.
    x = tf.reshape(x, [-1, width, height, self._total_value_depth])
    x = tf.transpose(a=x, perm=[0, 2, 1, 3])
    # Height axis striding.
    if self._strides > 1:
      x = x[:, ::self._strides, :, :]

    # Reshape the height axis to the batch dimension.
    _, strided_height, _, _ = x.get_shape().as_list()
    x = tf.reshape(x, [-1, width, self._total_value_depth])
    x = self._width_axis(x, training=training)
    # Reshape back to a 4D tensor.
    x = tf.reshape(x, [-1, strided_height, width, self._total_value_depth])
    # Width axis striding.
    if self._strides > 1:
      x = x[:, :, ::self._strides, :]

    return x


class GlobalAttention2D(tf.keras.layers.Layer):
  """A 2D global attention layer."""

  def __init__(self,
               strides=1,
               filters=512,
               name='attention',
               key_expansion=1,
               value_expansion=2,
               query_shape=(129, 129),
               memory_flange=(32, 32),
               double_global_attention=False,
               **kwargs):
    """Initializes a GlobalAttention2D layer.

    Args:
      strides: An integer, the stride for the output, usually 1 or 2.
      filters: An integer, the base number of channels for the layer.
      name: A string, the name of the attention layer.
      key_expansion: A float, the channel expansion ratio for keys.
      value_expansion: A float, the channel expansion ratio for values.
      query_shape: An integer, the maximum query shape for both the height axis
        and the width axis.
      memory_flange: An integer list of length 2. The memory flange for the
        height axis and the width axis.
      double_global_attention: A boolean, whether to use two global attention
        layers. Two global attention layers match the parameter count to a
        seqentially applied height and width axial attention layer.
      **kwargs: A dictionary of keyword arguments passed to height-axis,
        width-axis, and 2D global AxialAttention.

    Returns:
      output: A [batch, strided height, strided width, output_channels] tensor.

    Raises:
      ValueError: If relative positional encoding is enforced in kwargs.
    """
    if any([kwargs.get('use_query_rpe_similarity', False),
            kwargs.get('use_key_rpe_similarity', False),
            kwargs.get('retrieve_value_rpe', False)]):
      raise ValueError('GlobalAttention2D does not support relative positional '
                       'encodings.')

    super(GlobalAttention2D, self).__init__(name=name)
    total_key_depth = int(round(filters * key_expansion))
    total_value_depth = int(round(filters * value_expansion))
    self._strides = strides
    self._double_global_attention = double_global_attention
    self._total_key_depth = total_key_depth
    self._total_value_depth = total_value_depth

    # Global attention does not support relative positional encodings.
    kwargs['use_query_rpe_similarity'] = False
    kwargs['use_key_rpe_similarity'] = False
    kwargs['retrieve_value_rpe'] = False
    self._kwargs = kwargs

  def build(self, input_shape):
    """Builds global attention layers according to the 4D input_shape."""
    _, height, width, _ = input_shape
    # Implement 2D global attention as 1D axial-attention by flattening the 2D
    # inputs into 1D. We also disable the relative positional encodings in
    # axial attention, so that only content-based attention is used. The query
    # shape is set to height * width, so that the axial attention is global.
    self._global = AxialAttention(
        total_key_depth=self._total_key_depth,
        total_value_depth=self._total_value_depth,
        query_shape=height*width,
        memory_flange=0,
        name='global',
        **self._kwargs)

    # Use two global attention layers in one residual block. This option
    # ensures that global attention models have similar number of layers and
    # parameters as axial-attention models.
    if self._double_global_attention:
      self._global2 = AxialAttention(
          total_key_depth=self._total_key_depth,
          total_value_depth=self._total_value_depth,
          query_shape=height*width,
          memory_flange=0,
          name='global2',
          **self._kwargs)

  def call(self, inputs, training=False):
    """Performs a forward pass.

    Args:
      inputs: An input [batch, height, width, channel] tensor.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      output: An output [batch, strided_height, strided_width,
        filters * value_expansion] tensor.
    """
    _, height, width, channel = inputs.get_shape().as_list()

    # Reshape the inputs so that the attention is global 2D.
    x = tf.reshape(inputs, [-1, height * width, channel])

    # Implement 2D global attention as 1D axial-attention by flattening the 2D
    # inputs into 1D. We also disable the relative positional encodings in
    # axial attention, so that only content-based attention is used.
    x = self._global(x, training=training)

    # Use two global attention layers in one residual block. This option
    # ensures that global attention models have the same number of layers and
    # parameters as axial-attention models.
    if self._double_global_attention:
      x = self._global2(x, training=training)
    x = tf.reshape(x, [-1, height, width, self._total_value_depth])
    if self._strides > 1:
      x = x[:, ::self._strides, ::self._strides, :]

    return x
