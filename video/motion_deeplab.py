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

"""This file contains the Motion-DeepLab architecture."""

import functools
from typing import Any, Dict, Text, Tuple

from absl import logging
import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import builder
from deeplab2.model import utils
from deeplab2.model.post_processor import motion_deeplab
from deeplab2.model.post_processor import post_processor_builder


class MotionDeepLab(tf.keras.Model):
  """This class represents the Motion-DeepLab meta architecture.

  This class is the basis of the Motion-DeepLab architecture. This Model can be
  used for Video Panoptic Segmentation or Segmenting and Tracking Every Pixel
  (STEP).
  """

  def __init__(self,
               config: config_pb2.ExperimentOptions,
               dataset_descriptor: dataset.DatasetDescriptor):
    """Initializes a Motion-DeepLab architecture.

    Args:
      config: A config_pb2.ExperimentOptions configuration.
      dataset_descriptor: A dataset.DatasetDescriptor.
    """
    super(MotionDeepLab, self).__init__(name='MotionDeepLab')

    if config.trainer_options.solver_options.use_sync_batchnorm:
      logging.info('Synchronized Batchnorm is used.')
      bn_layer = functools.partial(
          tf.keras.layers.experimental.SyncBatchNormalization,
          momentum=config.trainer_options.solver_options.batchnorm_momentum,
          epsilon=config.trainer_options.solver_options.batchnorm_epsilon)
    else:
      logging.info('Standard (unsynchronized) Batchnorm is used.')
      bn_layer = functools.partial(
          tf.keras.layers.BatchNormalization,
          momentum=config.trainer_options.solver_options.batchnorm_momentum,
          epsilon=config.trainer_options.solver_options.batchnorm_epsilon)

    self._encoder = builder.create_encoder(
        config.model_options.backbone, bn_layer,
        conv_kernel_weight_decay=(
            config.trainer_options.solver_options.weight_decay))

    self._decoder = builder.create_decoder(config.model_options, bn_layer,
                                           dataset_descriptor.ignore_label)

    self._prev_center_prediction = tf.Variable(
        0.0,
        trainable=False,
        validate_shape=False,
        shape=tf.TensorShape(None),
        dtype=tf.float32,
        name='prev_prediction_buffer')
    self._prev_center_list = tf.Variable(
        tf.zeros((0, 5), dtype=tf.int32),
        trainable=False,
        validate_shape=False,
        shape=tf.TensorShape(None),
        name='prev_prediction_list')
    self._next_tracking_id = tf.Variable(
        1,
        trainable=False,
        validate_shape=False,
        dtype=tf.int32,
        name='next+_tracking_id')

    self._post_processor = post_processor_builder.get_post_processor(
        config, dataset_descriptor)
    self._render_fn = functools.partial(
        motion_deeplab.render_panoptic_map_as_heatmap,
        sigma=8,
        label_divisor=dataset_descriptor.panoptic_label_divisor,
        void_label=dataset_descriptor.ignore_label)
    self._track_fn = functools.partial(
        motion_deeplab.assign_instances_to_previous_tracks,
        label_divisor=dataset_descriptor.panoptic_label_divisor)
    # The ASPP pooling size is always set to train crop size, which is found to
    # be experimentally better.
    pool_size = config.train_dataset_options.crop_size
    output_stride = float(config.model_options.backbone.output_stride)
    pool_size = tuple(
        utils.scale_mutable_sequence(pool_size, 1.0 / output_stride))
    logging.info('Setting pooling size to %s', pool_size)
    self.set_pool_size(pool_size)

  def call(self, input_tensor: tf.Tensor, training=False) -> Dict[Text, Any]:
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels]. The input tensor should contain batches of RGB images.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).

    Returns:
      A dictionary containing the results of the specified DeepLab architecture.
      The results are bilinearly upsampled to input size before returning.
    """
    if not training:
      # During evaluation, we add the previous predicted heatmap as 7th input
      # channel (cf. during training, we use groundtruth heatmap).
      input_tensor = self._add_previous_heatmap_to_input(input_tensor)
    # Normalize the input in the same way as Inception. We normalize it outside
    # the encoder so that we can extend encoders to different backbones without
    # copying the normalization to each encoder. We normalize it after data
    # preprocessing because it is faster on TPUs than on host CPUs. The
    # normalization should not increase TPU memory consumption because it does
    # not require gradient.
    input_tensor = input_tensor / 127.5 - 1.0
    # Get the static spatial shape of the input tensor.
    _, input_h, input_w, _ = input_tensor.get_shape().as_list()

    pred = self._decoder(
        self._encoder(input_tensor, training=training), training=training)
    result_dict = dict()
    for key, value in pred.items():
      if (key == common.PRED_OFFSET_MAP_KEY or
          key == common.PRED_FRAME_OFFSET_MAP_KEY):
        result_dict[key] = utils.resize_and_rescale_offsets(
            value, [input_h, input_w])
      else:
        result_dict[key] = utils.resize_bilinear(
            value, [input_h, input_w])

    # Change the semantic logits to probabilities with softmax.
    result_dict[common.PRED_SEMANTIC_PROBS_KEY] = tf.nn.softmax(
        result_dict[common.PRED_SEMANTIC_LOGITS_KEY])
    if not training:
      result_dict.update(self._post_processor(result_dict))

      next_heatmap, next_centers = self._render_fn(
          result_dict[common.PRED_PANOPTIC_KEY])
      panoptic_map, next_centers, next_id = self._track_fn(
          self._prev_center_list.value(),
          next_centers,
          next_heatmap,
          result_dict[common.PRED_FRAME_OFFSET_MAP_KEY],
          result_dict[common.PRED_PANOPTIC_KEY],
          self._next_tracking_id.value()
      )

      result_dict[common.PRED_PANOPTIC_KEY] = panoptic_map
      self._next_tracking_id.assign(next_id)
      self._prev_center_prediction.assign(
          tf.expand_dims(next_heatmap, axis=3, name='expand_prev_centermap'))
      self._prev_center_list.assign(next_centers)

    if common.PRED_CENTER_HEATMAP_KEY in result_dict:
      result_dict[common.PRED_CENTER_HEATMAP_KEY] = tf.squeeze(
          result_dict[common.PRED_CENTER_HEATMAP_KEY], axis=3)
    return result_dict

  def _add_previous_heatmap_to_input(self, input_tensor: tf.Tensor
                                     ) -> tf.Tensor:
    frame1, frame2 = tf.split(input_tensor, [3, 3], axis=3)
    # We use a simple way to detect if the first frame of a sequence is being
    # processed. For the first frame, frame1 and frame2 are identical.
    if tf.reduce_all(tf.equal(frame1, frame2)):
      h = tf.shape(input_tensor)[1]
      w = tf.shape(input_tensor)[2]
      prev_center = tf.zeros((1, h, w, 1), dtype=tf.float32)
      self._prev_center_list.assign(tf.zeros((0, 5), dtype=tf.int32))
      self._next_tracking_id.assign(1)
    else:
      prev_center = self._prev_center_prediction
    output_tensor = tf.concat([frame1, frame2, prev_center], axis=3)
    output_tensor.set_shape([None, None, None, 7])
    return output_tensor

  def reset_pooling_layer(self):
    """Resets the ASPP pooling layer to global average pooling."""
    self._decoder.reset_pooling_layer()

  def set_pool_size(self, pool_size: Tuple[int, int]):
    """Sets the pooling size of the ASPP pooling layer.

    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layer.
    """
    self._decoder.set_pool_size(pool_size)

  @property
  def checkpoint_items(self) -> Dict[Text, Any]:
    items = dict(encoder=self._encoder)
    items.update(self._decoder.checkpoint_items)
    return items

  @property
  def auxiliary_output_number(self) -> int:
    # auxiliary_output_number is only supported in K-MaX meta, thus we hard
    # code it to 0 here.
    return 0
