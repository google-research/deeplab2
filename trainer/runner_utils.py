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

"""Utility functions for the trainer and evaluator runner."""

from typing import Any, Mapping, Union

import tensorflow as tf

from deeplab2 import config_pb2
from deeplab2.data import data_utils
from deeplab2.data import dataset
from deeplab2.data import multicamera_data_utils
from deeplab2.data import sample_generator
from deeplab2.data.dataloader import input_reader
from deeplab2.data.dataloader import multicamera_input_reader
from deeplab2.model.encoder import axial_resnet
from deeplab2.model.layers import axial_block_groups


def _load_tf_model_garden_vision_checkpoint(initial_checkpoint):
  # Determine whether the initial_checkpoint is trained by TensorFlow Model
  # Garden Vision trainer. This trainer applies a hardcoded prefix "backbone" to
  # DeepLab model variables that start with "_encoder".
  checkpoint_reader = tf.train.load_checkpoint(initial_checkpoint)
  variable_to_shape_map = checkpoint_reader.get_variable_to_shape_map()
  for variable in variable_to_shape_map:
    if variable.startswith('backbone/_encoder/'):
      return True
  return False


def maybe_load_checkpoint(initial_checkpoint: Union[str, None],
                          load_dict: Mapping[Any, Any]) -> None:
  """Maybe load a checkpoint.

  Args:
    initial_checkpoint: A string or None, specifying a path to a checkpoint.
    load_dict: A dictionary that defines what to load from the checkpoint.

  Raises:
    ValueError: If load_dict does not contain the 'encoder'.
  """
  if not initial_checkpoint:
    return

  if 'encoder' not in load_dict:
    raise ValueError('Load_dict should contain the encoder, but it is missing.')

  if tf.io.gfile.isdir(initial_checkpoint):
    initial_checkpoint = tf.train.latest_checkpoint(initial_checkpoint)

  if _load_tf_model_garden_vision_checkpoint(initial_checkpoint):
    checkpoint = tf.train.Checkpoint(
        backbone=tf.train.Checkpoint(
            _encoder=load_dict['encoder']))
  else:
    checkpoint = tf.train.Checkpoint(**load_dict)
  status = checkpoint.read(initial_checkpoint)
  # Motion-DeepLab models require nontrivial_match, as the input channels for
  # the first convolution change.
  status.expect_partial().assert_nontrivial_match()


def create_dataset(dataset_config: config_pb2.DatasetOptions,
                   is_training: bool,
                   only_semantic_annotations: bool = False):
  """Creates a tf.data.Dataset from the configuration.

  Args:
    dataset_config: A dataset_pb2.DatasetOptions configuration.
    is_training: A flag specifying if the dataset is used for training.
    only_semantic_annotations: A flag specifying if only semantic segmentation
      ground-truth should be generated.

  Returns:
    A tf.data.Dataset.
  """
  dataset_info = dataset.MAP_NAME_TO_DATASET_INFO[dataset_config.dataset]
  if dataset_info.camera_names is None:
    decoder = data_utils.SegmentationDecoder(
        is_panoptic_dataset=True,
        is_video_dataset=dataset_info.is_video_dataset,
        is_depth_dataset=dataset_info.is_depth_dataset,
        use_two_frames=dataset_config.use_two_frames,
        use_next_frame=dataset_config.use_next_frame,
        decode_groundtruth_label=dataset_config.decode_groundtruth_label)
  else:
    decoder = multicamera_data_utils.MultiCameraSegmentationDecoder(
        is_panoptic_dataset=True,
        is_video_dataset=dataset_info.is_video_dataset,
        is_depth_dataset=dataset_info.is_depth_dataset,
        cameras_to_use=dataset_info.camera_names,
        use_next_frame=dataset_config.use_next_frame,
        decode_groundtruth_label=dataset_config.decode_groundtruth_label,
    )

  focus_small_instances = None
  if dataset_config.increase_small_instance_weights:
    focus_small_instances = {
        'threshold': dataset_config.small_instance_threshold,
        'weight': dataset_config.small_instance_weight,
    }

  augmentation_options = dataset_config.augmentations
  if augmentation_options.HasField('panoptic_copy_paste'):
    panoptic_copy_paste_options = augmentation_options.panoptic_copy_paste
  else:
    panoptic_copy_paste_options = None
  generator_kwargs = dict(
      dataset_info=dataset_info._asdict(),
      is_training=is_training,
      crop_size=dataset_config.crop_size,
      min_resize_value=dataset_config.min_resize_value,
      max_resize_value=dataset_config.max_resize_value,
      resize_factor=dataset_config.resize_factor,
      min_scale_factor=augmentation_options.min_scale_factor,
      max_scale_factor=augmentation_options.max_scale_factor,
      scale_factor_step_size=augmentation_options.scale_factor_step_size,
      autoaugment_policy_name=augmentation_options.autoaugment_policy_name,
      only_semantic_annotations=only_semantic_annotations,
      thing_id_mask_annotations=dataset_config.thing_id_mask_annotations,
      max_thing_id=dataset_config.max_thing_id,
      sigma=dataset_config.sigma,
      focus_small_instances=focus_small_instances,
      panoptic_copy_paste_options=panoptic_copy_paste_options)
  reader_kwargs = dict(
      file_pattern=dataset_config.file_pattern,
      decoder_fn=decoder,
      use_panoptic_copy_paste=dataset_config.augmentations.HasField(
          'panoptic_copy_paste'),
      is_training=is_training,
  )
  if dataset_info.camera_names is None:
    generator = sample_generator.PanopticSampleGenerator(**generator_kwargs)
    reader = input_reader.InputReader(**reader_kwargs, generator_fn=generator)
  else:
    generator = sample_generator.MultiCameraPanopticSampleGenerator(
        **generator_kwargs)
    reader = multicamera_input_reader.MultiCameraInputReader(
        **reader_kwargs,
        camera_names=dataset_config.cameras_to_use,
        generator_fn=generator)

  return reader(dataset_config.batch_size)


def create_loss_metric_dict(loss_names, prefix='train_'):
  """Creates a loss metric dict.

  This function creates a metric for each loss name.

  Args:
    loss_names: A string list of N loss names.
    prefix: A string prefix, e.g., 'train_' or 'eval_'.

  Returns:
    loss_metric_dict: A dictionary of N tf.keras.metrics.Mean.
  """
  loss_metric_dict = {}
  for loss_name in loss_names:
    loss_metric = tf.keras.metrics.Mean(
        prefix + loss_name, dtype=tf.float32)
    loss_metric_dict[loss_name] = loss_metric
  return loss_metric_dict


def check_if_variable_in_backbone(
    variable, encoder_name, encoder_variable_names):
  """Determines whether a variable belongs to the pretrained backbone.

  The use case of this function could be to find all variables in the backbone,
  and then, we can use a smaller learning rate for them during training. For
  example, in MaX-DeepLab, we use 0.1x learning rate for the backbone. This is
  implemented by building a backbone optimizer (besides the base optimizer) for
  all variables that have been pretrained on a classification task. For other
  DeepLab variants, a smaller backbone learning rate is supported although it is
  not used by default.

  Args:
    variable: A tf.Variable, the variable to check.
    encoder_name: A string, the name of the DeepLab encoder.
    encoder_variable_names: A list of strings, all variable names of the DeepLab
      encoder.

  Returns:
    variable_in_backbone: A bool, whether the variable belongs to the backbone.
  """
  # Return false if the variable is not part of the encoder.
  if variable.name not in encoder_variable_names:
    return False
  # The variable is part of the encoder.
  # Return true if the encoder is not max_deeplab_s or max_deeplab_l.
  if encoder_name not in ('max_deeplab_s', 'max_deeplab_l'):
    return True
  # The variable is part of a max_deeplab encoder.
  # Return false for excluded keywords.
  if any([axial_block_groups.TRANSFORMER in variable.name,
          axial_resnet.EXTRA in variable.name,
          axial_resnet.MEMORY_FEATURE in variable.name]):
    return False
  return True
