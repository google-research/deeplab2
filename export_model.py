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

r"""Script to export deeplab model to saved model."""

import functools
from typing import Any, MutableMapping, Sequence, Text

from absl import app
from absl import flags
import tensorflow as tf

from google.protobuf import text_format
from deeplab2 import config_pb2
from deeplab2.data import dataset
from deeplab2.data.preprocessing import input_preprocessing
from deeplab2.model import utils
from deeplab2.trainer import train_lib


_FLAGS_EXPERIMENT_OPTION_PATH = flags.DEFINE_string(
    'experiment_option_path',
    default='',
    help='Path to the experiment option text proto.')

_FLAGS_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    default='',
    help='Path to the saved checkpoint.')

_FLAGS_OUTPUT_PATH = flags.DEFINE_string(
    'output_path',
    default='',
    help='Output directory path for the exported saved model.')

_FLAGS_MERGE_WITH_TF_OP = flags.DEFINE_boolean(
    'merge_with_tf_op',
    default=False,
    help='Whether to use customized TF op for merge semantic and instance '
    'predictions. Set it to True to reproduce the numbers as reported in '
    'paper, but the saved model would require specifically compiled TensorFlow '
    'to run.')


class DeepLabModule(tf.Module):
  """Class that runs DeepLab inference end-to-end."""

  def __init__(self, config: config_pb2.ExperimentOptions, ckpt_path: Text,
               use_tf_op: bool = False):
    super().__init__(name='DeepLabModule')

    dataset_options = config.eval_dataset_options
    dataset_name = dataset_options.dataset
    crop_height, crop_width = dataset_options.crop_size

    config.evaluator_options.merge_semantic_and_instance_with_tf_op = use_tf_op
    # Disable drop path and recompute grad as they are only used in training.
    config.model_options.backbone.drop_path_keep_prob = 1.0

    deeplab_model = train_lib.create_deeplab_model(
        config,
        dataset.MAP_NAME_TO_DATASET_INFO[dataset_name])
    meta_architecture = config.model_options.WhichOneof('meta_architecture')
    self._is_motion_deeplab = meta_architecture == 'motion_deeplab'
    self._is_vip_deeplab = meta_architecture == 'vip_deeplab'

    # For now we only support batch size of 1 for saved model.
    input_shape = train_lib.build_deeplab_model(
        deeplab_model, (crop_height, crop_width), batch_size=1)
    self._input_depth = input_shape[-1]

    checkpoint = tf.train.Checkpoint(**deeplab_model.checkpoint_items)
    # Not all saved variables (e.g. variables from optimizer) will be restored.
    # `expect_partial()` to suppress the warning.
    checkpoint.restore(ckpt_path).expect_partial()
    self._model = deeplab_model

    self._preprocess_fn = functools.partial(
        input_preprocessing.preprocess_image_and_label,
        label=None,
        crop_height=crop_height,
        crop_width=crop_width,
        prev_label=None,
        min_resize_value=dataset_options.min_resize_value,
        max_resize_value=dataset_options.max_resize_value,
        resize_factor=dataset_options.resize_factor,
        is_training=False)

  def get_input_spec(self):
    """Returns TensorSpec of input tensor needed for inference."""
    # We expect a single 3D, uint8 tensor with shape [height, width, channels].
    return tf.TensorSpec(shape=[None, None, self._input_depth], dtype=tf.uint8)

  @tf.function
  def __call__(self, input_tensor: tf.Tensor) -> MutableMapping[Text, Any]:
    """Performs a forward pass.

    Args:
      input_tensor: An uint8 input tensor of type tf.Tensor with shape [height,
        width, channels].

    Returns:
      A dictionary containing the results of the specified DeepLab architecture.
      The results are bilinearly upsampled to input size before returning.
    """
    input_size = [tf.shape(input_tensor)[0], tf.shape(input_tensor)[1]]

    if self._is_motion_deeplab or self._is_vip_deeplab:
      # For motion deeplab / vip-deeplab, split the input tensor to current
      # and previous / next frame before preprocessing, and re-assemble them.
      image_1, image_2 = tf.split(input_tensor, 2, axis=2)
      (resized_image, processed_image_1, _, processed_image_2,
       _, _) = self._preprocess_fn(image=image_1, prev_image=image_2)
      processed_image = tf.concat(
          [processed_image_1, processed_image_2], axis=2)
    else:
      (resized_image, processed_image, _, _, _, _) = self._preprocess_fn(
          image=input_tensor)

    resized_size = tf.shape(resized_image)[0:2]
    # Making input tensor to 4D to fit model input requirements.
    outputs = self._model(tf.expand_dims(processed_image, 0), training=False)
    # We only undo-preprocess for those defined in tuples in model/utils.py.
    return utils.undo_preprocessing(outputs, resized_size,
                                    input_size)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  config = config_pb2.ExperimentOptions()
  with tf.io.gfile.GFile(_FLAGS_EXPERIMENT_OPTION_PATH.value, 'r') as f:
    text_format.Parse(f.read(), config)

  module = DeepLabModule(
      config, _FLAGS_CKPT_PATH.value, _FLAGS_MERGE_WITH_TF_OP.value)

  signatures = module.__call__.get_concrete_function(module.get_input_spec())
  tf.saved_model.save(
      module, _FLAGS_OUTPUT_PATH.value, signatures=signatures)


if __name__ == '__main__':
  app.run(main)
