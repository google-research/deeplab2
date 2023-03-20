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

"""AutoAugment utility file.

Please cite or refer to the following papers:
- Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le.
"Autoaugment: Learning augmentation policies from data." In CVPR, 2019.

- Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le.
"Randaugment: Practical automated data augmentation with a reduced search
space." In CVPR, 2020.
"""

import inspect

import tensorflow as tf

from deeplab2.data.preprocessing import autoaugment_policy


# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
_MAX_LEVEL = 10.


def blend(image1, image2, factor):
  """Blends image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  """
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1, tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)


def invert(image):
  """Inverts the image pixels."""
  image = tf.convert_to_tensor(image)
  return 255 - image


def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def contrast(image, factor):
  """Equivalent of PIL Contrast."""
  degenerate = tf.image.rgb_to_grayscale(image)
  # Cast before calling tf.histogram.
  degenerate = tf.cast(degenerate, tf.int32)

  # Compute the grayscale histogram, then compute the mean pixel value,
  # and create a constant image size of that value.  Use that as the
  # blending degenerate target of the original image.
  hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
  mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
  degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
  return blend(degenerate, image, factor)


def brightness(image, factor):
  """Equivalent of PIL Brightness."""
  degenerate = tf.zeros_like(image)
  return blend(degenerate, image, factor)


def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def sharpness(image, factor):
  """Implements Sharpness function from PIL using TF ops."""
  orig_image = image
  image = tf.cast(image, tf.float32)
  # Make image 4D for conv operation.
  image = tf.expand_dims(image, 0)
  # SMOOTH PIL Kernel.
  kernel = tf.constant(
      [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
      shape=[3, 3, 1, 1]) / 13.
  # Tile across channel dimension.
  kernel = tf.tile(kernel, [1, 1, 3, 1])
  strides = [1, 1, 1, 1]
  degenerate = tf.nn.depthwise_conv2d(
      image, kernel, strides, padding='VALID', dilations=[1, 1])
  degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
  degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])

  # For the borders of the resulting image, fill in the values of the
  # original image.
  mask = tf.ones_like(degenerate)
  padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
  padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
  result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)

  # Blend the final result.
  return blend(result, orig_image, factor)


def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Invert': invert,
    'Posterize': posterize,
    'Solarize': solarize,
    'Color': color,
    'Contrast': contrast,
    'Brightness': brightness,
    'Sharpness': sharpness,
}


def _enhance_level_to_arg(level):
  return ((level/_MAX_LEVEL) * 1.8 + 0.1,)


def level_to_arg():
  return {
      'AutoContrast':
          lambda level: (),
      'Equalize':
          lambda level: (),
      'Invert':
          lambda level: (),
      'Posterize': lambda level: (int((level/_MAX_LEVEL) * 4),),
      'Solarize': lambda level: (int((level/_MAX_LEVEL) * 256),),
      'Color':
          _enhance_level_to_arg,
      'Contrast':
          _enhance_level_to_arg,
      'Brightness':
          _enhance_level_to_arg,
      'Sharpness':
          _enhance_level_to_arg,
  }


def label_wrapper(func):
  """Adds a label function argument to func and returns unchanged label."""
  def wrapper(images, label, *args, **kwargs):
    return func(images, *args, **kwargs), label
  return wrapper


def _parse_policy_info(name, prob, level, replace_value, ignore_label):
  """Returns the function corresponding to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  args = level_to_arg()[name](level)

  if 'prob' in inspect.getfullargspec(func)[0]:
    args = tuple([prob] + list(args))

  # Add in replace arg if it is required for the function that is being called.
  if 'replace' in inspect.getfullargspec(func)[0]:
    # Make sure ignore_label is also in the argument.
    assert 'ignore_label' in inspect.getfullargspec(func)[0]
    # Make sure replace is the second from last argument
    assert 'replace' == inspect.getfullargspec(func)[0][-2]
    # Make sure ignore_label is the final argument
    assert 'ignore_label' == inspect.getfullargspec(func)[0][-1]
    args = tuple(list(args) + [replace_value, ignore_label])

  # Add label as the second positional argument for the function if it does
  # not already exist.
  if 'label' not in inspect.getfullargspec(func)[0]:
    func = label_wrapper(func)
  return (func, prob, args)


def _apply_func_with_prob(func, image, args, prob, label):
  """Apply `func` to image w/ `args` as input with probability `prob`."""
  assert isinstance(args, tuple)
  assert 'label' == inspect.getfullargspec(func)[0][1]

  # If prob is a function argument, then this randomness is being handled
  # inside the function, so make sure it is always called.
  if 'prob' in inspect.getfullargspec(func)[0]:
    prob = 1.0

  # Apply the function with probability `prob`.
  should_apply_op = tf.cast(
      tf.floor(tf.random.uniform([], dtype=tf.float32) + prob), tf.bool)
  augmented_image, augmented_label = tf.cond(
      should_apply_op,
      lambda: func(image, label, *args),
      lambda: (image, label))
  return augmented_image, augmented_label


def select_and_apply_random_policy(policies, image, label):
  """Select a random policy from `policies` and apply it to `image`."""
  policy_to_select = tf.random.uniform([], maxval=len(policies), dtype=tf.int32)
  # Note that using tf.case instead of tf.conds would result in significantly
  # larger graphs and would even break export for some larger policies.
  for (i, policy) in enumerate(policies):
    image, label = tf.cond(
        tf.equal(i, policy_to_select),
        lambda selected_policy=policy: selected_policy(image, label),
        lambda: (image, label))
  return (image, label)


def build_and_apply_autoaugment_policy(policies, image, label, ignore_label):
  """Builds a policy from the given policies passed in and applies to image.

  Args:
    policies: list of lists of tuples in the form `(func, prob, level)`, `func`
      is a string name of the augmentation function, `prob` is the probability
      of applying the `func` operation, `level` is the input argument for
      `func`.
    image: tf.Tensor that the resulting policy will be applied to.
    label: tf.Tensor that the resulting policy will be applied to.
    ignore_label: The label value which will be ignored for training and
      evaluation.

  Returns:
    A version of image that now has data augmentation applied to it based on
    the `policies` pass into the function. Additionally, returns bboxes if
    a value for them is passed in that is not None
  """
  replace_value = [128, 128, 128]

  # func is the string name of the augmentation function, prob is the
  # probability of applying the operation and level is the parameter associated
  # with the tf op.

  # tf_policies are functions that take in an image and return an augmented
  # image.
  tf_policies = []
  for policy in policies:
    tf_policy = []
    # Link string name to the correct python function and make sure the correct
    # argument is passed into that function.
    for policy_info in policy:
      policy_info = (
          list(policy_info) + [replace_value, ignore_label])

      tf_policy.append(_parse_policy_info(*policy_info))
    # Now build the tf policy that will apply the augmentation procedue
    # on image.
    def make_final_policy(tf_policy_):
      def final_policy(image_, label_):
        for func, prob, args in tf_policy_:
          image_, label_ = _apply_func_with_prob(
              func, image_, args, prob, label_)
        return image_, label_
      return final_policy
    tf_policies.append(make_final_policy(tf_policy))

  augmented_images, augmented_label = select_and_apply_random_policy(
      tf_policies, image, label)
  # If no bounding boxes were specified, then just return the images.
  return (augmented_images, augmented_label)


def distort_image_with_autoaugment(image,
                                   label,
                                   ignore_label,
                                   augmentation_name=None):
  """Applies the AutoAugment policy to `image` and `label`.

  Args:
    image: `Tensor` of shape [height, width, 3] representing an image.
    label: `Tensor` of shape [height, width, 1] representing a label.
    ignore_label: The label value which will be ignored for training and
      evaluation.
    augmentation_name: The name of the AutoAugment policy to use. See
      autoaugment_policy.py for available_policies.

  Returns:
    A tuple containing the augmented versions of `image` and `label`.

  Raises:
    ValueError: If the augmentation_name is not in available_policies.
  """
  if augmentation_name:
    available_policies = autoaugment_policy.available_policies
    if augmentation_name not in available_policies:
      raise ValueError(
          'Invalid augmentation_name: {}'.format(augmentation_name))
    policy = available_policies[augmentation_name]
    return build_and_apply_autoaugment_policy(
        policy, image, label, ignore_label)
  return image, label
