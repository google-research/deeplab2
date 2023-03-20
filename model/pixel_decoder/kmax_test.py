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

"""Tests for kMaX pixel decoder."""

import tensorflow as tf

from deeplab2.model.pixel_decoder import kmax


class KMaXPixelDecoderTest(tf.test.TestCase):

  def test_model_output_shape(self):
    model = kmax.KMaXPixelDecoder(name='kmax_pixel_decoder')
    output = model({
        'stage1': tf.keras.Input(shape=(321, 321, 64)),
        'stage2': tf.keras.Input(shape=(161, 161, 128)),
        'stage3': tf.keras.Input(shape=(81, 81, 256)),
        'stage4': tf.keras.Input(shape=(41, 41, 512)),
        'stage5': tf.keras.Input(shape=(21, 21, 1024)),
    })

    self.assertListEqual(output['decoder_stage1'].get_shape().as_list(),
                         [None, 21, 21, 2048])
    self.assertListEqual(output['decoder_stage2'].get_shape().as_list(),
                         [None, 41, 41, 1024])
    self.assertListEqual(output['decoder_stage3'].get_shape().as_list(),
                         [None, 81, 81, 512])
    self.assertListEqual(output['decoder_output'].get_shape().as_list(),
                         [None, 161, 161, 256])


if __name__ == '__main__':
  tf.test.main()
