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

"""This file contains the DeepLab meta architecture."""
import collections
import functools
from typing import Any, Dict, Text, Tuple

from absl import logging
import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import builder
from deeplab2.model import utils
from deeplab2.model.post_processor import post_processor_builder

_OFFSET_OUTPUT = 'offset'


class DeepLab(tf.keras.Model):
  """This class represents the DeepLab meta architecture.

  This class supports four architectures of the DeepLab family: DeepLab V3,
  DeepLab V3+, Panoptic-DeepLab, and MaX-DeepLab. The exact architecture must be
  defined during initialization.
  """

  def __init__(self,
               config: config_pb2.ExperimentOptions,
               dataset_descriptor: dataset.DatasetDescriptor):
    """Initializes a DeepLab architecture.

    Args:
      config: A config_pb2.ExperimentOptions configuration.
      dataset_descriptor: A dataset.DatasetDescriptor.

    Raises:
      ValueError: If MaX-DeepLab is used with multi-scale inference.
    """
    super(DeepLab, self).__init__(name='DeepLab')

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

    # Divide weight decay by 2 to match the implementation of tf.nn.l2_loss. In
    # this way, we allow our users to use a normal weight decay (e.g., 1e-4 for
    # ResNet variants) in the config textproto. Then, we pass the adjusted
    # weight decay (e.g., 5e-5 for ResNets) to keras in order to exactly match
    # the commonly used tf.nn.l2_loss in TF1. References:
    # https://github.com/tensorflow/models/blob/68ee72ae785274156b9e943df4145b257cd78b32/official/vision/beta/tasks/image_classification.py#L41
    # https://www.tensorflow.org/api_docs/python/tf/keras/regularizers/l2
    # https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    self._encoder = builder.create_encoder(
        config.model_options.backbone, bn_layer,
        conv_kernel_weight_decay=(
            config.trainer_options.solver_options.weight_decay / 2))

    self._decoder = builder.create_decoder(
        config.model_options, bn_layer, dataset_descriptor.ignore_label)

    self._is_max_deeplab = (
        config.model_options.WhichOneof('meta_architecture') == 'max_deeplab')
    self._post_processor = post_processor_builder.get_post_processor(
        config, dataset_descriptor)

    # The ASPP pooling size is always set to train crop size, which is found to
    # be experimentally better.
    pool_size = config.train_dataset_options.crop_size
    output_stride = float(config.model_options.backbone.output_stride)
    pool_size = tuple(
        utils.scale_mutable_sequence(pool_size, 1.0 / output_stride))
    logging.info('Setting pooling size to %s', pool_size)
    self.set_pool_size(pool_size)

    # Variables for multi-scale inference.
    self._add_flipped_images = config.evaluator_options.add_flipped_images
    if not config.evaluator_options.eval_scales:
      self._eval_scales = [1.0]
    else:
      self._eval_scales = config.evaluator_options.eval_scales
    if self._is_max_deeplab and (
        self._add_flipped_images or len(self._eval_scales) > 1):
      raise ValueError(
          'MaX-DeepLab does not support multi-scale inference yet.')

  def call(self,
           input_tensor: tf.Tensor,
           training: bool = False) -> Dict[Text, Any]:
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
    # Normalize the input in the same way as Inception. We normalize it outside
    # the encoder so that we can extend encoders to different backbones without
    # copying the normalization to each encoder. We normalize it after data
    # preprocessing because it is faster on TPUs than on host CPUs. The
    # normalization should not increase TPU memory consumption because it does
    # not require gradient.
    input_tensor = input_tensor / 127.5 - 1.0
    # Get the static spatial shape of the input tensor.
    _, input_h, input_w, _ = input_tensor.get_shape().as_list()
    if training:
      result_dict = self._decoder(
          self._encoder(input_tensor, training=training), training=training)
      result_dict = self._resize_predictions(
          result_dict,
          target_h=input_h,
          target_w=input_w)
    else:
      result_dict = collections.defaultdict(list)
      # Evaluation mode where one could perform multi-scale inference.
      scale_1_pool_size = self.get_pool_size()
      logging.info('Eval with scales %s', self._eval_scales)
      for eval_scale in self._eval_scales:
        # Get the scaled images/pool_size for each scale.
        scaled_images, scaled_pool_size = (
            self._scale_images_and_pool_size(
                input_tensor, list(scale_1_pool_size), eval_scale))
        # Update the ASPP pool size for different eval scales.
        self.set_pool_size(tuple(scaled_pool_size))
        logging.info('Eval scale %s; setting pooling size to %s',
                     eval_scale, scaled_pool_size)
        pred_dict = self._decoder(
            self._encoder(scaled_images, training=training), training=training)
        # MaX-DeepLab skips this resizing and upsamples the mask outputs in
        # self._post_processor.
        pred_dict = self._resize_predictions(
            pred_dict,
            target_h=input_h,
            target_w=input_w)
        # Change the semantic logits to probabilities with softmax. Note
        # one should remove semantic logits for faster inference. We still
        # keep them since they will be used to compute evaluation loss.
        pred_dict[common.PRED_SEMANTIC_PROBS_KEY] = tf.nn.softmax(
            pred_dict[common.PRED_SEMANTIC_LOGITS_KEY])
        # Store the predictions from each scale.
        for output_type, output_value in pred_dict.items():
          result_dict[output_type].append(output_value)
        if self._add_flipped_images:
          pred_dict_reverse = self._decoder(
              self._encoder(tf.reverse(scaled_images, [2]), training=training),
              training=training)
          pred_dict_reverse = self._resize_predictions(
              pred_dict_reverse,
              target_h=input_h,
              target_w=input_w,
              reverse=True)
          # Change the semantic logits to probabilities with softmax.
          pred_dict_reverse[common.PRED_SEMANTIC_PROBS_KEY] = tf.nn.softmax(
              pred_dict_reverse[common.PRED_SEMANTIC_LOGITS_KEY])
          # Store the predictions from each scale.
          for output_type, output_value in pred_dict_reverse.items():
            result_dict[output_type].append(output_value)
      # Set back the pool_size for scale 1.0, the original setting.
      self.set_pool_size(tuple(scale_1_pool_size))
      # Average results across scales.
      for output_type, output_value in result_dict.items():
        result_dict[output_type] = tf.reduce_mean(
            tf.stack(output_value, axis=0), axis=0)
      # Post-process the results.
      result_dict.update(self._post_processor(result_dict))

    if common.PRED_CENTER_HEATMAP_KEY in result_dict:
      result_dict[common.PRED_CENTER_HEATMAP_KEY] = tf.squeeze(
          result_dict[common.PRED_CENTER_HEATMAP_KEY], axis=3)
    return result_dict

  def reset_pooling_layer(self):
    """Resets the ASPP pooling layer to global average pooling."""
    self._decoder.reset_pooling_layer()

  def set_pool_size(self, pool_size: Tuple[int, int]):
    """Sets the pooling size of the ASPP pooling layer.

    Args:
      pool_size: A tuple specifying the pooling size of the ASPP pooling layer.
    """
    self._decoder.set_pool_size(pool_size)

  def get_pool_size(self):
    return self._decoder.get_pool_size()

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

  def _resize_predictions(self, result_dict, target_h, target_w, reverse=False):
    """Resizes predictions to the target height and width.

    This function resizes the items in the result_dict to the target height and
    width. The items are optionally reversed w.r.t width if `reverse` is True.

    Args:
      result_dict: A dictionary storing prediction results to be resized.
      target_h: An integer, the target height.
      target_w: An integer, the target width.
      reverse: A boolean, reversing the prediction result w.r.t. width.

    Returns:
      Resized (or optionally reversed) result_dict.
    """
    # The default MaX-DeepLab paper does not upsample any output during training
    # in order to save GPU/TPU memory, but upsampling might lead to better
    # performance.
    if self._is_max_deeplab:
      return result_dict
    for key, value in result_dict.items():
      if reverse:
        value = tf.reverse(value, [2])
        # Special care to offsets: need to flip x-offsets.
        if _OFFSET_OUTPUT in key:
          offset_y, offset_x = tf.split(
              value=value, num_or_size_splits=2, axis=3)
          offset_x *= -1
          value = tf.concat([offset_y, offset_x], 3)
      if _OFFSET_OUTPUT in key:
        result_dict[key] = utils.resize_and_rescale_offsets(
            value, [target_h, target_w])
      else:
        result_dict[key] = utils.resize_bilinear(
            value, [target_h, target_w])
    return result_dict

  def _scale_images_and_pool_size(self, images, pool_size, scale):
    """Scales images and pool_size w.r.t. scale.

    Args:
      images: An input tensor with shape [batch, height, width, 3].
      pool_size: A list with two elements, specifying the pooling size
        of ASPP pooling layer.
      scale: A float, used to scale the input images and pool_size.

    Returns:
      Scaled images, and pool_size.
    """
    if scale == 1.0:
      scaled_images = images
      scaled_pool_size = pool_size
    else:
      image_size = images.get_shape().as_list()[1:3]
      scaled_image_size = utils.scale_mutable_sequence(image_size, scale)
      scaled_images = utils.resize_bilinear(images, scaled_image_size)
      scaled_pool_size = [None, None]
      if pool_size != [None, None]:
        scaled_pool_size = utils.scale_mutable_sequence(pool_size, scale)
    return scaled_images, scaled_pool_size
