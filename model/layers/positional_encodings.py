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

"""Implements relative [1, 2, 3] and global [3, 4] positional encodings.

Our Axial-Deeplab [1] proposes position-sensitive self-attention which uses
relative positional encodings for query, key, and value.

[1] Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
    ECCV 2020 Spotlight.
      Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
      Liang-Chieh Chen.
[2] Self-Attention with Relative Position Representations, NAACL 2018.
      Peter Shaw, Jakob Uszkoreit, Ashish Vaswani.
[3] Tensor2Tensor for Neural Machine Translation, arXiv 2018,
    http://arxiv.org/abs/1803.07416.
      Ashish Vaswani, Samy Bengio, Eugene Brevdo, Francois Chollet,
      Aidan N. Gomez, Stephan Gouws, Llion Jones, Łukasz Kaiser,
      Nal Kalchbrenner, Niki Parmar, Ryan Sepassi, Noam Shazeer,
      Jakob Uszkoreit.
[4] Attention Is All You Need, NeurIPS 2017.
      Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
      Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.
[5] An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale,
    ICLR 2021.
      Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn,
      Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer,
      Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
"""

from absl import logging
import tensorflow as tf

# MAX_SPAN defines the maximum shape of positional encoding. It is set as a
# large constant so that we can easily load and use models with global or
# different local spans, but it should not be too large so that it takes a
# reasonable amount of memory. The value 255 is larger than almost all span
# choices (e.g. 65 for local attention, 129, 193, etc.) so 255 is large enough.
# 257 will be a good choice for gpu, but 255 is more efficient on TPU which pads
# tensors to 128x.
MAX_SPAN = 255


def _compute_relative_distance_matrix(query_length, key_length):
  """Computes a relative distance matrix between queries and keys.

  We assume that the queries and the keys are centered, i.e.,
  key_length = memory_flange + query_length + memory_flange.

  The function is based on the _generate_relative_positions_matrix function in
  common_attention.py of tensor2tensor codebase:
  https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1670

  Args:
    query_length: An integer, the length of queries.
    key_length: An integer, the length of keys.

  Returns:
    distance_matrix: A [query_length, key_length] tensor.

  Raises:
    ValueError: If (key_length - query_length) is odd, i.e., the assumption does
      not hold.
  """
  if (key_length - query_length) % 2:
    raise ValueError('Key_length should be query_length + 2 * memory_flange.')
  key_index = tf.range(key_length)
  query_index = tf.range(query_length) + (key_length - query_length) // 2
  distance_matrix = key_index[None, :] - query_index[:, None]
  # Shift the distance_matrix so that it is >= 0. Each entry of the
  # distance_matrix distance will index a relative positional embedding.
  distance_matrix = distance_matrix + MAX_SPAN - 1
  if query_length + (key_length - query_length) // 2 > MAX_SPAN:
    logging.warn(
        'Axial attention span is larger than MAX_SPAN. In this case, we use a '
        'single shared embedding for all positions beyond this relative '
        'distance. Please make sure, this behavior is intended.')
    distance_matrix = tf.clip_by_value(distance_matrix, 0, MAX_SPAN * 2 - 2)
  return distance_matrix


class RelativePositionalEncoding(tf.keras.layers.Layer):
  """Generates relative positional encoding.

  The function is based on the _generate_relative_positions_embeddings function
  in common_attention.py of tensor2tensor codebase:
  https://github.com/tensorflow/tensor2tensor/blob/5623deb79cfcd28f8f8c5463b58b5bd76a81fd0d/tensor2tensor/layers/common_attention.py#L1691
  """

  def __init__(self, query_length, key_length, depth, name,
               initialization_std=1.0, conv_kernel_weight_decay=0.0):
    """Initializes a relative position encoding layer.

    Args:
      query_length: An integer, the length of queries.
      key_length: An integer, the length of keys.
      depth: An integer, the number of embedding channels per head.
      name: A string, the name of the embedding.
      initialization_std: A float, the initialization std for the embedding.
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.

    Returns:
      output: A [query, key, depth] tensor, the relative positional
        encodings for each head and each query-key-pair.
    """
    super(RelativePositionalEncoding, self).__init__(name=name)
    self._initializer = tf.keras.initializers.TruncatedNormal(
        stddev=initialization_std)
    self._regularizer = tf.keras.regularizers.l2(conv_kernel_weight_decay)

    self._relative_distance_matrix = _compute_relative_distance_matrix(
        query_length, key_length)
    self._embedding_shape = (MAX_SPAN * 2 - 1, depth)

  def build(self, input_shape):
    """Builds the embedding weight."""
    del input_shape
    self._embeddings = self.add_weight(
        shape=self._embedding_shape,
        initializer=self._initializer, trainable=True,
        name='embeddings',
        regularizer=self._regularizer)

  def call(self, inputs):
    """A forward pass that gathers the relative positional encoding."""
    del inputs
    # Gather the embeddings according to the relative distances.
    return tf.gather(self._embeddings, self._relative_distance_matrix)


class AddAbsolutePositionalEncoding(tf.keras.layers.Layer):
  """Adds a learnable absolute positional encoding to the input feature.

  Supports both 1D and 2D versions of the positional encoding: (1) 1D positional
  encoding represents each row index with an embedding, and represents each
  column index with another embedding. This results in a total of (height +
  width) learnable embedding vectors. (2) 2D positional encoding adds
  independent embeddings to each input grid position. This choice uses a total
  of (height * width) learnable embedding vectors.
  """

  def __init__(self, name, positional_encoding_type=None,
               bn_layer=tf.keras.layers.BatchNormalization,
               conv_kernel_weight_decay=0.0):
    """Initializes an AddAbsolutePositionEmbedding layer.

    Args:
      name: A string specifying the name of the layer.
      positional_encoding_type: A string, type of the positional encoding.
        Support '2D', '1D', 'none', and None. The feature is returned as is if
        positional_encoding_type is 'none' or None.
      bn_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      conv_kernel_weight_decay: A float, the weight decay for convolution
        kernels.

    Raises:
      ValueError: If positional_encoding_type is not one of '1D', '2D', 'none',
        and None.
    """
    super(AddAbsolutePositionalEncoding, self).__init__(name=name)
    if not any([positional_encoding_type is None,
                positional_encoding_type.lower() == 'none',
                positional_encoding_type.lower() == '2d',
                positional_encoding_type.lower() == '1d']):
      raise ValueError(positional_encoding_type + ' is not supported.')
    self._positional_encoding_type = positional_encoding_type
    # This initialization std is tuned for global attention, but it does not
    # seem to be a sensitive hyper-parameter, since we use batch norm on the
    # positional encodings.
    self._initializer = tf.keras.initializers.TruncatedNormal(stddev=0.2)
    self._kernel_regularizer = tf.keras.regularizers.l2(
        conv_kernel_weight_decay)
    self._bn_layer = bn_layer

  def build(self, input_shape):
    """Builds the layer weights whose shape depends on the 4D input shape."""
    _, height, width, channel = input_shape
    if self._positional_encoding_type.lower() == '2d':
      self._embeddings = self.add_weight(
          shape=(1, height, width, channel),
          initializer=self._initializer, trainable=True,
          name='embeddings',
          regularizer=self._kernel_regularizer)
      self._batch_norm = self._bn_layer(axis=-1, name='batch_norm')
    elif self._positional_encoding_type.lower() == '1d':
      # Generate separable positional encodings for the height axis and the
      # width axis.
      self._height_axis_embeddings = self.add_weight(
          shape=(1, height, 1, channel),
          initializer=self._initializer, trainable=True,
          name='height_axis_embeddings',
          regularizer=self._kernel_regularizer)
      self._height_axis_batch_norm = self._bn_layer(
          axis=-1, name='height_axis_batch_norm')
      self._width_axis_embeddings = self.add_weight(
          shape=(1, height, 1, channel),
          initializer=self._initializer, trainable=True,
          name='width_axis_embeddings',
          regularizer=self._kernel_regularizer)
      self._width_axis_batch_norm = self._bn_layer(
          axis=-1, name='width_axis_batch_norm')

  def call(self, features, training=False):
    """Performs a forward pass.

    Args:
      features: An input [batch, height, width, channels] tensor.
      training: A boolean, whether the model is in training mode.

    Returns:
      output: The sum of the input feature and learnable positional encodings.
    """
    if (self._positional_encoding_type is None or
        self._positional_encoding_type.lower() == 'none'):
      return features
    elif self._positional_encoding_type.lower() == '2d':
      positional_encoding = self._batch_norm(self._embeddings,
                                             training=training)
    elif self._positional_encoding_type.lower() == '1d':
      height_axis_positional_encoding = self._height_axis_batch_norm(
          self._height_axis_embeddings, training=training)
      width_axis_positional_encoding = self._width_axis_batch_norm(
          self._width_axis_embeddings, training=training)
      positional_encoding = (height_axis_positional_encoding +
                             width_axis_positional_encoding)
    return features + positional_encoding
