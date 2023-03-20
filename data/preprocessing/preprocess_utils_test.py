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

"""Tests for preprocess_utils."""
import numpy as np
import tensorflow as tf

from deeplab2.data.preprocessing import preprocess_utils


class PreprocessUtilsTest(tf.test.TestCase):

  def testNoFlipWhenProbIsZero(self):
    numpy_image = np.dstack([[[5., 6.],
                              [9., 0.]],
                             [[4., 3.],
                              [3., 5.]]])
    image = tf.convert_to_tensor(numpy_image)

    actual, is_flipped = preprocess_utils.flip_dim([image], prob=0, dim=0)
    self.assertAllEqual(numpy_image, actual)
    self.assertFalse(is_flipped)
    actual, is_flipped = preprocess_utils.flip_dim([image], prob=0, dim=1)
    self.assertAllEqual(numpy_image, actual)
    self.assertFalse(is_flipped)
    actual, is_flipped = preprocess_utils.flip_dim([image], prob=0, dim=2)
    self.assertAllEqual(numpy_image, actual)
    self.assertFalse(is_flipped)

  def testFlipWhenProbIsOne(self):
    numpy_image = np.dstack([[[5., 6.],
                              [9., 0.]],
                             [[4., 3.],
                              [3., 5.]]])
    dim0_flipped = np.dstack([[[9., 0.],
                               [5., 6.]],
                              [[3., 5.],
                               [4., 3.]]])
    dim1_flipped = np.dstack([[[6., 5.],
                               [0., 9.]],
                              [[3., 4.],
                               [5., 3.]]])
    dim2_flipped = np.dstack([[[4., 3.],
                               [3., 5.]],
                              [[5., 6.],
                               [9., 0.]]])
    image = tf.convert_to_tensor(numpy_image)

    actual, is_flipped = preprocess_utils.flip_dim([image], prob=1, dim=0)
    self.assertAllEqual(dim0_flipped, actual)
    self.assertTrue(is_flipped)
    actual, is_flipped = preprocess_utils.flip_dim([image], prob=1, dim=1)
    self.assertAllEqual(dim1_flipped, actual)
    self.assertTrue(is_flipped)
    actual, is_flipped = preprocess_utils.flip_dim([image], prob=1, dim=2)
    self.assertAllEqual(dim2_flipped, actual)
    self.assertTrue(is_flipped)

  def testFlipMultipleImagesConsistentlyWhenProbIsOne(self):
    numpy_image = np.dstack([[[5., 6.],
                              [9., 0.]],
                             [[4., 3.],
                              [3., 5.]]])
    numpy_label = np.dstack([[[0., 1.],
                              [2., 3.]]])
    image_dim1_flipped = np.dstack([[[6., 5.],
                                     [0., 9.]],
                                    [[3., 4.],
                                     [5., 3.]]])
    label_dim1_flipped = np.dstack([[[1., 0.],
                                     [3., 2.]]])
    image = tf.convert_to_tensor(numpy_image)
    label = tf.convert_to_tensor(numpy_label)

    image, label, is_flipped = preprocess_utils.flip_dim(
        [image, label], prob=1, dim=1)
    self.assertAllEqual(image_dim1_flipped, image)
    self.assertAllEqual(label_dim1_flipped, label)
    self.assertTrue(is_flipped)

  def testReturnRandomFlipsOnMultipleEvals(self):
    numpy_image = np.dstack([[[5., 6.],
                              [9., 0.]],
                             [[4., 3.],
                              [3., 5.]]])
    dim1_flipped = np.dstack([[[6., 5.],
                               [0., 9.]],
                              [[3., 4.],
                               [5., 3.]]])
    image = tf.convert_to_tensor(numpy_image)
    original_image, not_flipped = preprocess_utils.flip_dim(
        [image], prob=0, dim=1)
    flip_image, is_flipped = preprocess_utils.flip_dim(
        [image], prob=1.0, dim=1)
    self.assertAllEqual(numpy_image, original_image)
    self.assertFalse(not_flipped)
    self.assertAllEqual(dim1_flipped, flip_image)
    self.assertTrue(is_flipped)

  def testReturnCorrectCropOfSingleImage(self):
    np.random.seed(0)

    height, width = 10, 20
    image = np.random.randint(0, 256, size=(height, width, 3))

    crop_height, crop_width = 2, 4

    [cropped] = preprocess_utils.random_crop([tf.convert_to_tensor(image)],
                                             crop_height,
                                             crop_width)

    # Ensure we can find the cropped image in the original:
    is_found = False
    for x in range(0, width - crop_width + 1):
      for y in range(0, height - crop_height + 1):
        if np.isclose(image[y:y+crop_height, x:x+crop_width, :],
                      cropped).all():
          is_found = True
          break

    self.assertTrue(is_found)

  def testRandomCropMaintainsNumberOfChannels(self):
    np.random.seed(0)

    crop_height, crop_width = 10, 20
    image = np.random.randint(0, 256, size=(100, 200, 3))

    tf.random.set_seed(37)
    [cropped] = preprocess_utils.random_crop(
        [tf.convert_to_tensor(image)], crop_height, crop_width)

    self.assertListEqual(cropped.shape.as_list(), [crop_height, crop_width, 3])

  def testReturnDifferentCropAreasOnTwoEvals(self):
    tf.random.set_seed(0)

    crop_height, crop_width = 2, 3
    image = np.random.randint(0, 256, size=(100, 200, 3))
    [cropped0] = preprocess_utils.random_crop(
        [tf.convert_to_tensor(image)], crop_height, crop_width)
    [cropped1] = preprocess_utils.random_crop(
        [tf.convert_to_tensor(image)], crop_height, crop_width)

    self.assertFalse(np.isclose(cropped0.numpy(), cropped1.numpy()).all())

  def testReturnConsistenCropsOfImagesInTheList(self):
    tf.random.set_seed(0)

    height, width = 10, 20
    crop_height, crop_width = 2, 3
    labels = np.linspace(0, height * width-1, height * width)
    labels = labels.reshape((height, width, 1))
    image = np.tile(labels, (1, 1, 3))

    [cropped_image, cropped_label] = preprocess_utils.random_crop(
        [tf.convert_to_tensor(image), tf.convert_to_tensor(labels)],
        crop_height, crop_width)

    for i in range(3):
      self.assertAllEqual(cropped_image[:, :, i], tf.squeeze(cropped_label))

  def testDieOnRandomCropWhenImagesWithDifferentWidth(self):
    crop_height, crop_width = 2, 3
    image1 = tf.convert_to_tensor(np.random.rand(4, 5, 3))
    image2 = tf.convert_to_tensor(np.random.rand(4, 6, 1))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = preprocess_utils.random_crop([image1, image2], crop_height,
                                       crop_width)

  def testDieOnRandomCropWhenImagesWithDifferentHeight(self):
    crop_height, crop_width = 2, 3
    image1 = tf.convert_to_tensor(np.random.rand(4, 5, 3))
    image2 = tf.convert_to_tensor(np.random.rand(5, 5, 1))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = preprocess_utils.random_crop([image1, image2], crop_height,
                                       crop_width)

  def testDieOnRandomCropWhenCropSizeIsGreaterThanImage(self):
    crop_height, crop_width = 5, 9
    image1 = tf.convert_to_tensor(np.random.rand(4, 5, 3))
    image2 = tf.convert_to_tensor(np.random.rand(4, 5, 1))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      _ = preprocess_utils.random_crop([image1, image2], crop_height,
                                       crop_width)

  def testRandomScaleFitsInRange(self):
    scale_value = preprocess_utils.get_random_scale(1., 2., 0.)
    self.assertGreaterEqual(scale_value, 1.)
    self.assertLessEqual(scale_value, 2.)

  def testDeterminedRandomScaleReturnsNumber(self):
    scale = preprocess_utils.get_random_scale(1., 1., 0.)
    self.assertEqual(scale, 1.)

  def testResizeTensorsToRange(self):
    test_shapes = [[60, 40],
                   [15, 30],
                   [15, 50]]
    min_size = 50
    max_size = 100
    factor = None
    expected_shape_list = [(75, 50, 3),
                           (50, 100, 3),
                           (30, 100, 3)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=None,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True)
      self.assertEqual(new_tensor_list[0].shape, expected_shape_list[i])

  def testResizeTensorsToRangeWithFactor(self):
    test_shapes = [[60, 40],
                   [15, 30],
                   [15, 50]]
    min_size = 50
    max_size = 98
    factor = 8
    expected_image_shape_list = [(81, 57, 3),
                                 (49, 97, 3),
                                 (33, 97, 3)]
    expected_label_shape_list = [(81, 57, 1),
                                 (49, 97, 1),
                                 (33, 97, 1)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      label = tf.random.normal([test_shape[0], test_shape[1], 1])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=label,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True)
      self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])
      self.assertEqual(new_tensor_list[1].shape, expected_label_shape_list[i])

  def testResizeTensorsToRangeWithSimilarMinMaxSizes(self):
    test_shapes = [[60, 40],
                   [15, 30],
                   [15, 50]]
    # Values set so that one of the side = 97.
    min_size = 96
    max_size = 98
    factor = 8
    expected_image_shape_list = [(97, 65, 3),
                                 (49, 97, 3),
                                 (33, 97, 3)]
    expected_label_shape_list = [(97, 65, 1),
                                 (49, 97, 1),
                                 (33, 97, 1)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      label = tf.random.normal([test_shape[0], test_shape[1], 1])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=label,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True)
      self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])
      self.assertEqual(new_tensor_list[1].shape, expected_label_shape_list[i])

  def testResizeTensorsToRangeWithEqualMaxSize(self):
    test_shapes = [[97, 38],
                   [96, 97]]
    # Make max_size equal to the larger value of test_shapes.
    min_size = 97
    max_size = 97
    factor = 8
    expected_image_shape_list = [(97, 41, 3),
                                 (97, 97, 3)]
    expected_label_shape_list = [(97, 41, 1),
                                 (97, 97, 1)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      label = tf.random.normal([test_shape[0], test_shape[1], 1])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=label,
          min_size=min_size,
          max_size=max_size,
          factor=factor,
          align_corners=True)
      self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])
      self.assertEqual(new_tensor_list[1].shape, expected_label_shape_list[i])

  def testResizeTensorsToRangeWithPotentialErrorInTFCeil(self):
    test_shape = [3936, 5248]
    # Make max_size equal to the larger value of test_shapes.
    min_size = 1441
    max_size = 1441
    factor = 16
    expected_image_shape = (1089, 1441, 3)
    expected_label_shape = (1089, 1441, 1)
    image = tf.random.normal([test_shape[0], test_shape[1], 3])
    label = tf.random.normal([test_shape[0], test_shape[1], 1])
    new_tensor_list = preprocess_utils.resize_to_range(
        image=image,
        label=label,
        min_size=min_size,
        max_size=max_size,
        factor=factor,
        align_corners=True)
    self.assertEqual(new_tensor_list[0].shape, expected_image_shape)
    self.assertEqual(new_tensor_list[1].shape, expected_label_shape)

  def testResizeTensorWithOnlyMaxSize(self):
    test_shapes = [[97, 38],
                   [96, 18]]

    max_size = (97, 28)
    # Since the second test shape already fits max size, do nothing.
    expected_image_shape_list = [(71, 28, 3),
                                 (96, 18, 3)]
    for i, test_shape in enumerate(test_shapes):
      image = tf.random.normal([test_shape[0], test_shape[1], 3])
      new_tensor_list = preprocess_utils.resize_to_range(
          image=image,
          label=None,
          min_size=None,
          max_size=max_size,
          align_corners=True)
      self.assertEqual(new_tensor_list[0].shape, expected_image_shape_list[i])


if __name__ == '__main__':
  tf.test.main()
