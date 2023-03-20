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

"""Tests of model exports for axial_resnet_instances."""

import os

from absl import flags
from absl.testing import parameterized
import tensorflow as tf

from deeplab2.model.encoder import axial_resnet_instances

FLAGS = flags.FLAGS


class ModelExportTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ('resnet50',),
      ('resnet50_beta',),
      ('max_deeplab_s_backbone',),
      ('max_deeplab_l_backbone',),
      ('axial_resnet_s',),
      ('axial_resnet_l',),
      ('axial_deeplab_s',),
      ('axial_deeplab_l',),
      ('swidernet',),
      ('axial_swidernet',),
      )
  def test_model_export(self, model_name):
    model = axial_resnet_instances.get_model(
        model_name,
        output_stride=16,
        backbone_layer_multiplier=1.0,
        bn_layer=tf.keras.layers.BatchNormalization,
        conv_kernel_weight_decay=0.0001,
        # Test with small models only.
        num_blocks=[2, 2, 2, 2],
        # Disable drop path as it is not compatible with model exporting.
        block_group_config={'drop_path_keep_prob': 1.0})
    model(tf.keras.Input([257, 257, 3], batch_size=1), training=False)
    export_dir = os.path.join(
        FLAGS.test_tmpdir, 'test_model_export', model_name)
    model.save(export_dir)


if __name__ == '__main__':
  tf.test.main()
