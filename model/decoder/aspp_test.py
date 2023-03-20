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

"""Tests for aspp."""
import tensorflow as tf

from deeplab2.model.decoder import aspp
from deeplab2.utils import test_utils


class AsppTest(tf.test.TestCase):

  def test_aspp_pool_error(self):
    pool = aspp.ASPPPool(output_channels=64, name='')

    # Should pass without an error.
    pool.set_pool_size((None, None))

    with self.assertRaises(ValueError):
      # Should raise an error.
      pool.set_pool_size((2, None))

  def test_aspp_conv_atrous_rate_shape(self):
    atrous_rates = [2, 6, 12, 18]
    for rate in atrous_rates:
      conv = aspp.ASPPConv(output_channels=64, atrous_rate=rate, name='')
      input_tensor = tf.random.uniform(shape=(2, 12, 12, 3))

      output = conv(input_tensor)
      expected_shape = [2, 12, 12, 64]
      self.assertListEqual(output.shape.as_list(), expected_shape)

  def test_aspp_conv_non_negative(self):
    conv = aspp.ASPPConv(output_channels=12, atrous_rate=2, name='')
    input_tensor = tf.random.uniform(shape=(2, 17, 17, 3))

    output = conv(input_tensor)
    self.assertTrue((output.numpy() >= 0.0).all())

  def test_aspp_pool_shape(self):
    pool = aspp.ASPPPool(output_channels=64, name='')
    input_tensor = tf.random.uniform(shape=(2, 12, 12, 3))

    output = pool(input_tensor)
    expected_shape = [2, 12, 12, 64]
    self.assertListEqual(output.shape.as_list(), expected_shape)

  def test_aspp_pool_non_negative(self):
    pool = aspp.ASPPPool(output_channels=12, name='')
    input_tensor = tf.random.uniform(shape=(2, 17, 17, 3))

    output = pool(input_tensor)
    self.assertTrue((output.numpy() >= 0.0).all())

  def test_aspp_wrong_atrous_rate(self):
    with self.assertRaises(ValueError):
      _ = aspp.ASPP(output_channels=64, atrous_rates=[1, 2, 3, 4])

  @test_utils.test_all_strategies
  def test_aspp_shape(self, strategy):
    with strategy.scope():
      for bn_layer in test_utils.NORMALIZATION_LAYERS:
        aspp_layer = aspp.ASPP(
            output_channels=64, atrous_rates=[6, 12, 18], bn_layer=bn_layer)
        input_tensor = tf.random.uniform(shape=(2, 32, 32, 3))

        output = aspp_layer(input_tensor)
        expected_shape = [2, 32, 32, 64]
        self.assertListEqual(output.shape.as_list(), expected_shape)

  def test_aspp_non_negative(self):
    aspp_layer = aspp.ASPP(output_channels=32, atrous_rates=[4, 8, 16])
    input_tensor = tf.random.uniform(shape=(2, 32, 32, 3))

    output = aspp_layer(input_tensor)
    self.assertTrue((output.numpy() >= 0.0).all())

if __name__ == '__main__':
  tf.test.main()
