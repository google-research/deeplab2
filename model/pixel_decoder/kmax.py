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

"""Pixel Decoder used in k-means Mask Transformer.

The pixel decoder in the k-means Mask Transformer (kMaX-DeepLab) [1] employs a
simple decoder structure, similar to CMT-DeepLab [2] and MaX-DeepLab-S [3].

We support using axial-block [4] and bottleneck-block [5] in the decoder, along
with skip connections from the pixel encoder (i.e., backbone). When
self-attention operations are used (e.g., axial-blocks), it is equivalent to
incorporating the transformer encoder to the pixel decoder.

[1] k-means Mask Transformer, ECCV 2022.
      Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
      Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

[2] CMT-DeepLab: Clustering Mask Transformers for Panoptic Segmentation,
    CVPR 2022.
      Qihang Yu, Huiyu Wang, Dahun Kim, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
      Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

[3] MaX-DeepLab: End-to-End Panoptic Segmentation with Mask Transformers,
    CVPR 2021.
      Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

[4] Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
    ECCV 2020.
      Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
      Liang-Chieh Chen.

[5] Deep residual learning for image recognition.
    CVPR 2016.
      Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
"""

import functools

import tensorflow as tf

from deeplab2.model.layers import axial_block_groups
from deeplab2.model.layers import resized_fuse

axial_block = functools.partial(
    axial_block_groups.BlockGroup,
    original_resnet_stride=1,
    original_resnet_input_stride=32,
    use_axial_beyond_stride=16,
    use_transformer_beyond_stride=0,
    drop_path_keep_prob=1.0,
    activation='gelu',
    axial_use_recompute_grad=False)

bottleneck_block = functools.partial(
    axial_block_groups.BlockGroup,
    original_resnet_stride=1,
    original_resnet_input_stride=32,
    use_axial_beyond_stride=0,
    use_transformer_beyond_stride=0,
    drop_path_keep_prob=1.0,
    activation='gelu')


class KMaXPixelDecoder(tf.keras.Model):
  """Pixel Decoder for kMaX.

  The pixel decoder in the k-means Mask Transformer (kMaX) employs a simple
  decoder structure, similar to MaX-DeepLab-S model. We support using
  axial-block and bottleneck-block in the decoder, along with skip connections
  from the pixel encoder (i.e., backbone). When self-attention operations are
  used (e.g., axial-blocks), it is equivalent to incorporating the transformer
  encoder to the pixel decoder.

  References:
    [1] k-means Mask Transformer, ECCV 2022.
          Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
          Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  def __init__(self,
               name,
               norm_layer=tf.keras.layers.BatchNormalization,
               dims=(512, 256, 128, 64),
               num_blocks=(1, 5, 1, 1),
               block_type=('axial', 'axial', 'bottleneck', 'bottleneck')):
    """Initializes a PixelDecoder for kMaX.

    Args:
      name: A string, the name of the model.
      norm_layer: An optional tf.keras.layers.Layer that computes the
        normalization (default: tf.keras.layers.BatchNormalization).
      dims: A list of integers specifying number of channels for each stage. The
        stage is counted from backwards, i.e., output stride 32 to output
        stride 4/2.
      num_blocks: A list of integers specifying number of blocks for each stage.
        The stage is counted from backwards, i.e., output stride 32 to output
        stride 4/2.
      block_type: A list of integers specifying the block type for each stage.
        'axial' for axial block, 'bottleneck' for bottleneck block. The
        stage is counted from backwards, i.e., output stride 32 to output
        stride 4/2.

    Raises:
      ValueError: If the length of dims, num_blocks, block_type are not equal to
        each other.
      ValueError: If num_stages is not 4 or 5.
    """
    super().__init__(name=name)

    num_stages = len(dims)
    if num_stages != len(num_blocks) or num_stages != len(block_type):
      raise ValueError('Number of stages should be equal in dims, num_blocks,'
                       ' and block_type!')
    # We have 4/5 stages for os32, 16, 8, 4, (2) respectively.
    if num_stages not in [4, 5]:
      raise ValueError('Number of stages should be 4 or 5!')

    # Define layer norm appended to encoder features at each output stride.
    self._backbone_norms = []
    for i in range(num_stages):
      self._backbone_norms.append(
          tf.keras.layers.LayerNormalization(epsilon=1e-6))

    # Define decoder blocks.
    self._stages = []
    for i in range(num_stages):
      if block_type[i].lower() == 'axial':
        block_group_fn = axial_block(filters=dims[i],
                                     num_blocks=num_blocks[i],
                                     name=f'kmax_pixeldecoder_{i}',
                                     bn_layer=norm_layer)
      elif block_type[i].lower() == 'bottleneck':
        block_group_fn = bottleneck_block(filters=dims[i],
                                          num_blocks=num_blocks[i],
                                          name=f'kmax_pixeldecoder_{i}',
                                          bn_layer=norm_layer)
      else:
        raise ValueError('Unsupported block_type!')

      self._stages.append(block_group_fn)

    self._num_stages = num_stages
    self._norm_layer = norm_layer
    self._dims = dims
    self._stage_to_backbone_endpoint = {0: 'stage5',
                                        1: 'stage4',
                                        2: 'stage3',
                                        3: 'stage2',
                                        4: 'stage1',}

  def build(self, input_shape_dict):
    # Define skip-connections.
    self._skip_connections = []
    # Stage5 does not need any skip-connection.
    for i in range(self._num_stages - 1):
      decoder_height, decoder_width = (
          input_shape_dict[self._stage_to_backbone_endpoint[i + 1]][-3:-1])
      skip_connection_fn = resized_fuse.ResizedFuse(
          name=f'kmax_pixel_decoder_skip_connection_{i}',
          height=decoder_height,
          width=decoder_width,
          num_channels=self._dims[i + 1],
          activation='gelu',
          bn_layer=self._norm_layer)
      self._skip_connections.append(skip_connection_fn)

  def call(self, endpoints, training=False):
    # Make a copy so that input argument will not be modified, per requirements
    # from exporting a saved model.
    endpoints = dict(endpoints)

    x = self._backbone_norms[0](
        endpoints[self._stage_to_backbone_endpoint[0]], training=training)
    for i in range(self._num_stages - 1):
      x, _ = self._stages[i]((x, None), training=training)
      endpoints['decoder_stage{}'.format(i + 1)] = x
      x = self._skip_connections[i]([
          x,
          self._backbone_norms[i + 1](
              endpoints[self._stage_to_backbone_endpoint[i + 1]],
              training=training)], training=training)

    x, _ = self._stages[-1]((x, None), training=training)
    endpoints['decoder_output'] = x
    return endpoints
