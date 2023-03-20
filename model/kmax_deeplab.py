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

"""This file contains the kMaX-DeepLab meta architecture."""
import functools
from typing import Any, Dict, Text

from absl import logging
import tensorflow as tf

from deeplab2 import common
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.model import builder
from deeplab2.model import utils
from deeplab2.model.decoder import max_deeplab
from deeplab2.model.post_processor import post_processor_builder


class KMaXDeepLab(tf.keras.Model):
  """This class represents the kMaX-DeepLab meta architecture."""

  def __init__(self,
               config: config_pb2.ExperimentOptions,
               dataset_descriptor: dataset.DatasetDescriptor):
    """Initializes a kMaX-DeepLab architecture.

    Args:
      config: A config_pb2.ExperimentOptions configuration.
      dataset_descriptor: A dataset.DatasetDescriptor.

    Raises:
      ValueError: If normalization type is not one of ['sync_batchnorm',
        'batchnorm', 'layernorm'].
    """
    super(KMaXDeepLab, self).__init__(name='kMaXDeepLab')

    # We hard code self._auxiliary_output_number for now.
    self._auxiliary_output_number = 6

    if config.trainer_options.solver_options.use_sync_batchnorm:
      logging.info('Synchronized Batchnorm is used.')
      norm_layer = functools.partial(
          tf.keras.layers.experimental.SyncBatchNormalization,
          momentum=config.trainer_options.solver_options.batchnorm_momentum,
          epsilon=config.trainer_options.solver_options.batchnorm_epsilon)
    else:
      logging.info('Standard (unsynchronized) Batchnorm is used.')
      norm_layer = functools.partial(
          tf.keras.layers.BatchNormalization,
          momentum=config.trainer_options.solver_options.batchnorm_momentum,
          epsilon=config.trainer_options.solver_options.batchnorm_epsilon)

    # The auxiliary predictor does not include the auxiliary semantic head.
    def auxiliary_predictor_func():
      return max_deeplab.MaXDeepLab(
          config.model_options.decoder,
          config.model_options.max_deeplab,
          ignore_label=dataset_descriptor.ignore_label,
          bn_layer=norm_layer,
          use_auxiliary_semantic_decoder=False,
          use_auxiliary_semantic_head=False,
          activation='gelu')

    pretrained_weights_path = None
    if config.model_options.backbone.HasField('pretrained_weights'):
      pretrained_weights_path = config.model_options.backbone.pretrained_weights
    self._pixel_encoder = builder.create_kmax_meta_pixel_encoder(
        pixel_encoder_name=self._infer_pixel_encoder_name_from_config(config),
        drop_path_keep_prob=config.model_options.backbone.drop_path_keep_prob,
        input_shape=list(config.train_dataset_options.crop_size) + [3],
        norm_layer=norm_layer,
        pretrained_weights_path=pretrained_weights_path)

    # Currently, some properties (e.g., high_resolution_output_stride,
    # num_mask_slots) of kMaX are hard-coded for simplicity.
    if 'cityscapes' in dataset_descriptor.dataset_name:
      high_resolution_output_stride = 2
      num_mask_slots = 256
      use_auxiliary_semantic_decoder = False
      transformer_decoder_drop_path_keep_prob = 0.8
    elif 'coco' in dataset_descriptor.dataset_name:
      high_resolution_output_stride = 4
      num_mask_slots = 128
      use_auxiliary_semantic_decoder = True
      transformer_decoder_drop_path_keep_prob = 0.8
    elif 'ade20k' in dataset_descriptor.dataset_name:
      high_resolution_output_stride = 4
      num_mask_slots = 128
      use_auxiliary_semantic_decoder = True
      transformer_decoder_drop_path_keep_prob = 1.0
    else:
      logging.info('kMaX configs for COCO is used for untested datasets.')
      high_resolution_output_stride = 4
      num_mask_slots = 128
      use_auxiliary_semantic_decoder = True
      transformer_decoder_drop_path_keep_prob = 0.8

    self._pixel_decoder = builder.create_kmax_meta_pixel_decoder(
        norm_layer=norm_layer,
        high_resolution_output_stride=high_resolution_output_stride)

    self._transformer_decoder = builder.create_kmax_meta_transformer_decoder(
        norm_layer=norm_layer,
        num_mask_slots=num_mask_slots,
        transformer_decoder_drop_path_keep_prob=(
            transformer_decoder_drop_path_keep_prob),
        auxiliary_predictor_func=auxiliary_predictor_func)

    # Notably, decoder/max_deeplab.MaXDeepLab is in fact the MaX-DeepLab
    # prediction head instead of its decoder.
    self._predictor = max_deeplab.MaXDeepLab(
        config.model_options.decoder,
        config.model_options.max_deeplab,
        ignore_label=dataset_descriptor.ignore_label,
        bn_layer=norm_layer,
        use_auxiliary_semantic_decoder=use_auxiliary_semantic_decoder,
        use_auxiliary_semantic_head=True,
        activation='gelu')

    self._post_processor = post_processor_builder.get_post_processor(
        config, dataset_descriptor)

  def _infer_pixel_encoder_name_from_config(self, config):
    return config.model_options.backbone.name.replace('kmax_', '')

  def _forward_step(self, input_tensor, training=False):
    encoder_pixel_feature = self._pixel_encoder(input_tensor, training=training)
    decoder_pixel_feature = self._pixel_decoder(
        encoder_pixel_feature, training=training)
    decoder_pixel_and_mask_feature = self._transformer_decoder(
        decoder_pixel_feature, training=training)
    pred_dict = self._predictor(decoder_pixel_and_mask_feature,
                                training=training)
    return pred_dict

  def _inference(self,
                 input_tensor: tf.Tensor,
                 training: bool = False,
                 post_processing: bool = True) -> Dict[Text, Any]:
    """Performs an (optionally multi-scale) inference pass."""
    pred_dict = self._forward_step(input_tensor, training=training)
    result_dict = {}
    for output_type, output_value in pred_dict.items():
      # We remove auxiliary outputs from final result_dict during inference.
      if output_type not in [common.PRED_AUXILIARY_OUTPUTS]:
        result_dict[output_type] = output_value
    # Post-process the results.
    if post_processing:
      result_dict.update(self._post_processor(result_dict))
    return result_dict

  def call(self,
           input_tensor: tf.Tensor,
           training: bool = False,
           post_processing: bool = True) -> Dict[Text, Any]:
    """Performs a forward pass.

    Args:
      input_tensor: An input tensor of type tf.Tensor with shape [batch, height,
        width, channels] of [batch, num_frames, heigh, width, channels]. The
        input tensor should contain batches of RGB images.
      training: A boolean flag indicating whether training behavior should be
        used (default: False).
      post_processing: A boolean flag indicating whether post processing
        (self._post_processor) should be used during testing (default: True).

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
    if training:
      result_dict = self._forward_step(input_tensor, training=training)
      result_dict = self._prepare_auxiliary_results(result_dict)
    else:
      result_dict = self._inference(input_tensor, training, post_processing)
    return result_dict

  @property
  def auxiliary_output_number(self) -> int:
    return self._auxiliary_output_number

  @property
  def checkpoint_items(self) -> Dict[Text, Any]:
    items = dict(encoder=self._pixel_encoder,
                 pixel_decoder=self._pixel_decoder,
                 transformer_decoder=self._transformer_decoder)
    items.update(self._predictor.checkpoint_items)
    return items

  def _prepare_auxiliary_results(self, result_dict):
    """Prepares and resizes auxiliary predictions to the target size.

    This function prepares auxiliary outputs for loss computation. It also
    resizes the PRED_PIXEL_SPACE_MASK_LOGITS_KEY and
    PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY in the result_dict to the target
    height and width.

    Args:
      result_dict: A dictionary storing prediction results to be resized.

    Returns:
      Prepared auxiliary_result_list.
    """
    target_h, target_w = (
        result_dict[
            common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY].get_shape().as_list()[1:3])
    auxiliary_result_list = list(result_dict[common.PRED_AUXILIARY_OUTPUTS])
    for auxiliary_idx in range(len(auxiliary_result_list)):
      auxiliary_result_dict = auxiliary_result_list[auxiliary_idx]
      for key, value in auxiliary_result_dict.items():
        if key in [common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY,
                   common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY]:
          auxiliary_result_dict[key] = utils.resize_bilinear(
              value, [target_h, target_w])
      auxiliary_result_list[auxiliary_idx] = auxiliary_result_dict
    result_dict[common.PRED_AUXILIARY_OUTPUTS] = tuple(auxiliary_result_list)
    return result_dict
