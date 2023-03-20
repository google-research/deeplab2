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

"""Transformer Decoder used in k-means Mask Transformer.

The transformer decoder in the k-means Mask Transformer (kMaX-DeepLab) [1]
employs the k-means cross attention, where an argmax is operated along the
cluster center dimension (instead of a softmax along the spatial dimension as
in the first original Mask Transformer [2]). The argmax operation is similar to
the k-means pixel-cluster assignment step (with a hard assignment). The cluster
centers are then updated by aggregating the pixel features based on the
pixel-cluster assignment (computed by their feature affinity), similar to the
k-means center-update step.

[1] k-means Mask Transformer, ECCV 2022.
      Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
      Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

[2] MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
    CVPR 2021.
      Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
"""
import functools
from typing import Callable, Optional, Tuple

import tensorflow as tf

from deeplab2 import common
from deeplab2.model.layers import axial_block_groups
from deeplab2.model.layers import convolutions

# A transformer decoder block with multi-head self-attention and single-head
# k-means cross-attention, as proposed in kMaX-DeepLab.
transformer_decoder_block = functools.partial(
    axial_block_groups.BlockGroup,
    num_blocks=1,
    # Note that the variable 'filters' is required by BlockGroup, and thus we
    # set filters = 128 (a random value), which does not affect the dual-path
    # transformer (i.e., changing this value will have no effect here).
    filters=128,
    original_resnet_stride=1,
    original_resnet_input_stride=32,
    use_transformer_beyond_stride=16,
    # The channels of dual-path transformer is controlled by
    # (128 * transformer_expansion).
    transformer_expansion=1.0,
    activation='gelu',
    # Disable the pixel2pixel attention.
    use_axial_block=False,
    dual_path_transformer_layer_config={
        'use_memory_self_attention': True,
        'use_memory2pixel_feedback_attention': False,
        'use_pixel2memory_feedback_attention': False,
        'use_kmeans_cross_attention': True,
    })


class KMaXTransformerDecoder(tf.keras.Model):
  """kMaX Transformer Decoder.

  The transformer decoder in the k-means Mask Transformer (kMaX) employs the
  k-means cross attention, where an argmax is operated along the cluster center
  dimension (instead of a softmax along the spatial dimension). The argmax
  operation is similar to the k-means pixel-cluster assignment step (with a hard
  assignment). The cluster centers are then updated by aggregating the pixel
  features based on the pixel-cluster assignment (computed by their feature
  affinity), similar to the k-means center-update step.

  References:
    [1] k-means Mask Transformer, ECCV 2022.
          Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
          Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  def __init__(self,
               name: str,
               auxiliary_predictor_func: Optional[Callable[[], tf.keras.Model]],
               norm_layer: Optional[Callable[
                   [],
                   tf.keras.layers.Layer]] = tf.keras.layers.BatchNormalization,
               num_blocks: Tuple[int, int, int] = (2, 2, 2),
               num_mask_slots: int = 128,
               transformer_decoder_drop_path_keep_prob: float = 1.0):
    """Initializes a KMaXTransformerDecoder.

    Args:
      name: A string, the name of the model.
      auxiliary_predictor_func: A callable function that returns an
        initialization of auxiliary predictor.
      norm_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      num_blocks: A list of three integers specifying number of blocks for
        each stage. The stage is counted backwards, i.e., from output stride
        32, 16, and 8.
      num_mask_slots: An integer, the number of mask slots that will be used.
      transformer_decoder_drop_path_keep_prob: A float, the drop-path keep prob
        for transformer decoder.
    Raises:
      ValueError: If the length of num_blocks is not 3.
    """
    super().__init__(name=name)

    if len(num_blocks) != 3:
      raise ValueError('Expect the length of num_blocks to be 3!')

    self._kmax_decoder = []
    for index, feature_output_stride in enumerate([32, 16, 8]):
      for i in range(num_blocks[index]):
        kmax_decoder_fn = transformer_decoder_block(
            name=f'kmax_transformer_decoder_os{feature_output_stride}_{i}',
            bn_layer=norm_layer,
            auxiliary_predictor_func=auxiliary_predictor_func,
            drop_path_keep_prob=transformer_decoder_drop_path_keep_prob)
        self._kmax_decoder.append(kmax_decoder_fn)

    self._cluster_centers = self.add_weight(
        name='cluster_centers',
        shape=(1, num_mask_slots, 256),
        initializer=tf.keras.initializers.TruncatedNormal(stddev=1.0),
        trainable=True)

    self._class_embedding_projection = convolutions.Conv1D(
        256,
        'class_embedding_projection',
        use_bias=False,
        use_bn=True,
        bn_layer=norm_layer,
        activation='gelu')

    self._mask_embedding_projection = convolutions.Conv1D(
        256,
        'mask_embedding_projection',
        use_bias=False,
        use_bn=True,
        bn_layer=norm_layer,
        activation='gelu')

    self._num_blocks = num_blocks
    self._num_mask_slots = num_mask_slots

  def _prepare_cluster_centers(self, input_tensor, training=False):
    batch_size = tf.shape(input_tensor)[0]
    cluster_centers = tf.tile(self._cluster_centers, [batch_size, 1, 1])
    return cluster_centers

  def call(self, endpoints, training=False):
    # Make a copy so that input argument will not be modified, per requirements
    # from exporting a saved model.
    endpoints = dict(endpoints)

    # Apply kMaX decoder on pixel features at output stride 32, 16, and 8
    # respectively to update the cluster centers.
    feature_dict = {
        32: endpoints['decoder_stage1'],
        16: endpoints['decoder_stage2'],
        8: endpoints['decoder_stage3']}
    cluster_centers = self._prepare_cluster_centers(feature_dict[32],
                                                    training=training)
    auxiliary_outputs = ()
    current_transformer_idx = 0
    for index, feature_output_stride in enumerate([32, 16, 8]):
      for _ in range(self._num_blocks[index]):
        (_, cluster_centers, auxiliary_outputs) = (
            self._kmax_decoder[current_transformer_idx](
                (feature_dict[feature_output_stride], cluster_centers,
                 auxiliary_outputs), training=training))
        current_transformer_idx += 1

    # Project cluster centers to mask embeddings and class embeddings.
    class_embeddings = self._class_embedding_projection(
        cluster_centers, training=training)
    mask_embeddings = self._mask_embedding_projection(
        cluster_centers, training=training)
    # Prepare endpoints for predictor.
    endpoints['transformer_class_feature'] = class_embeddings
    endpoints['transformer_mask_feature'] = mask_embeddings
    endpoints['feature_panoptic'] = endpoints['decoder_output']
    endpoints['feature_semantic'] = endpoints['stage5']
    endpoints[common.PRED_AUXILIARY_OUTPUTS] = auxiliary_outputs

    return endpoints
