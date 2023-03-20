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

"""This file contains code to get a sample from a dataset."""

import functools

import numpy as np
import tensorflow as tf

from deeplab2 import common
from deeplab2.data import dataset_utils
from deeplab2.data.preprocessing import input_preprocessing as preprocessing
from deeplab2.model import utils


def _compute_gaussian_from_std(sigma):
  """Computes the Gaussian and its size from a given standard deviation."""
  size = int(6 * sigma + 3)
  x = np.arange(size, dtype=float)
  y = x[:, np.newaxis]
  x0, y0 = 3 * sigma + 1, 3 * sigma + 1
  return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)), size


class PanopticSampleGenerator:
  """This class generates samples from images and labels."""

  def __init__(self,
               dataset_info,
               is_training,
               crop_size,
               min_resize_value=None,
               max_resize_value=None,
               resize_factor=None,
               min_scale_factor=1.,
               max_scale_factor=1.,
               scale_factor_step_size=0,
               autoaugment_policy_name=None,
               only_semantic_annotations=False,
               thing_id_mask_annotations=False,
               max_thing_id=128,
               sigma=8,
               focus_small_instances=None,
               panoptic_copy_paste_options=None):
    """Initializes the panoptic segmentation generator.

    Args:
      dataset_info: A dictionary with the following keys.
      - `name`: String, dataset name.
      - `ignore_label`: Integer, ignore label.
      - `class_has_instances_list`: A list of integers indicating which
        class has instance annotations.
      - `panoptic_label_divisor`: Integer, panoptic label divisor.
      - `num_classes`: Integer, number of classes.
      - `is_video_dataset`: Boolean, is video dataset or not.
      is_training: Boolean, is training mode or not.
      crop_size: Image crop size [height, width].
      min_resize_value: A 2-tuple of (height, width), desired minimum value
        after resize. If a single element is given, then height and width share
        the same value. None, empty or having 0 indicates no minimum value will
        be used.
      max_resize_value: A 2-tuple of (height, width), maximum allowed value
        after resize. If a single element is given, then height and width
        share the same value. None, empty or having 0 indicates no maximum
        value will be used.
      resize_factor: Resized dimensions are multiple of factor plus one.
      min_scale_factor: Minimum scale factor for random scale augmentation.
      max_scale_factor: Maximum scale factor for random scale augmentation.
      scale_factor_step_size: The step size from min scale factor to max scale
        factor. The input is randomly scaled based on the value of
        (min_scale_factor, max_scale_factor, scale_factor_step_size).
      autoaugment_policy_name: String, autoaugment policy name. See
        autoaugment_policy.py for available policies.
      only_semantic_annotations: An optional flag indicating whether the model
        needs only semantic annotations (default: False).
      thing_id_mask_annotations: An optional flag indicating whether the model
        needs thing_id_mask annotations. When `thing_id_mask_annotations` is
        True, we will additionally return mask annotation for each `thing`
        instance, encoded with a unique thing_id. This ground-truth annotation
        could be used to learn a better segmentation mask for each instance.
        `thing_id` indicates the number of unique thing-ID to each instance in
        an image, starting the counting from 0 (default: False).
      max_thing_id: The maximum number of possible thing instances per image. It
        is used together when thing_id_mask_annotations = True, representing the
        maximum thing ID encoded in the thing_id_mask. (default: 128).
      sigma: The standard deviation of the Gaussian used to encode the center
        keypoint (default: 8).
      focus_small_instances: An optional dict that defines how to deal with
        small instances (default: None):
        -`threshold`: An integer defining the threshold pixel number for an
          instance to be considered small.
        -`weight`: A number that defines the loss weight for small instances.
      panoptic_copy_paste_options: An optional dict that defines how to perform
        panoptic copy-paste augmentations.
    """
    self._dataset_info = dataset_info
    self._ignore_label = self._dataset_info['ignore_label']
    self._ignore_depth = self._dataset_info['ignore_depth']
    self._only_semantic_annotations = only_semantic_annotations
    self._sigma = sigma
    self._instance_area_threshold = 0
    self._small_instance_weight = 1.0
    self._thing_id_mask_annotations = thing_id_mask_annotations
    self._max_thing_id = max_thing_id
    self._is_training = is_training
    self._panoptic_copy_paste_options = panoptic_copy_paste_options

    preprocess_kwargs = {
        'crop_height': crop_size[0],
        'crop_width': crop_size[1],
        'min_resize_value': min_resize_value,
        'max_resize_value': max_resize_value,
        'resize_factor': resize_factor,
        'scale_factor_step_size': scale_factor_step_size,
        'autoaugment_policy_name': autoaugment_policy_name,
        'ignore_depth': self._ignore_depth,
        'is_training': self._is_training
    }

    self._preprocessing_fn = functools.partial(
        preprocessing.preprocess_image_and_label,
        min_scale_factor=min_scale_factor,
        max_scale_factor=max_scale_factor,
        ignore_label=self._ignore_label *
        self._dataset_info['panoptic_label_divisor'],
        **preprocess_kwargs)

    if panoptic_copy_paste_options:
      # For panoptic copy-paste, the ignore_label is set to
      # ignore_label * label_divisor + 1 instead of
      # ignore_label * label_divisor, so that padded pixels and void pixels can
      # be distinguished.
      self._panoptic_copy_paste_padding_value = (
          self._ignore_label * self._dataset_info['panoptic_label_divisor'] + 1)
      self._panoptic_copy_paste_preprocessing_fn = functools.partial(
          preprocessing.preprocess_image_and_label,
          min_scale_factor=min_scale_factor *
          panoptic_copy_paste_options.panoptic_copy_paste_scale,
          max_scale_factor=max_scale_factor *
          panoptic_copy_paste_options.panoptic_copy_paste_scale,
          ignore_label=self._panoptic_copy_paste_padding_value,
          **preprocess_kwargs)

    if focus_small_instances is not None:
      self._instance_area_threshold = focus_small_instances['threshold']
      self._small_instance_weight = focus_small_instances['weight']

    self._gaussian, self._gaussian_size = _compute_gaussian_from_std(
        self._sigma)
    self._gaussian = tf.cast(tf.reshape(self._gaussian, [-1]), tf.float32)

  def __call__(self, sample_dict, panoptic_copy_psate_sample_dict=None):
    """Gets a sample.

    When panoptic copy-paste augmentation is enabled, a tuple of sample_dict
    will be passed to this method, and this method will unzip the tuple.

    Args:
      sample_dict: A dictionary with the following keys and values:
      - `image`: A tensor of shape [image_height, image_width, 3].
      - `image_name`: String, image name.
      - `label`: A tensor of shape [label_height, label_width, 1] or None.
      - `height`: An integer specifying the height of the image.
      - `width`: An integer specifying the width of the image.
      - `sequence`: An optional string specifying the sequence name.
      - `prev_image`: An optional tensor of the same shape as `image`.
      - `prev_label`: An optional tensor of the same shape as `label`.
      - `next_image`: An optional next-frame tensor of the shape of `image`.
      - `next_label`: An optional next-frame tensor of the shape of `label`.
      - `depth`: An optional tensor of the same shape as `label`.
      panoptic_copy_psate_sample_dict: Additional sample_dict for panoptic
        copy-paste augmentation, it is None at default. It is a dictionary
        containing 'image' and 'label' if panoptic copy-paste is enabled.

    Returns:
      sample: A dictionary storing required data for panoptic segmentation.
    """
    if panoptic_copy_psate_sample_dict is not None:
      sample_dict['panoptic_copy_paste_image'] = (
          panoptic_copy_psate_sample_dict['image'])
      sample_dict['panoptic_copy_paste_label'] = (
          panoptic_copy_psate_sample_dict['label'])

    return self.call(**sample_dict)

  def call(self,
           image,
           image_name,
           label,
           height,
           width,
           sequence='',
           prev_image=None,
           prev_label=None,
           next_image=None,
           next_label=None,
           depth=None,
           panoptic_copy_paste_image=None,
           panoptic_copy_paste_label=None,
           frame_id=''):
    """Gets a sample.

    Args:
      image: A tensor of shape [image_height, image_width, 3].
      image_name: String, image name.
      label: A tensor of shape [label_height, label_width, 1] or None.
      height: An integer specifying the height of the image.
      width: An integer specifying the width of the image.
      sequence: An optional string specifying the sequence name.
      prev_image: An optional tensor of shape [image_height, image_width, 3].
      prev_label: An optional tensor of shape [label_height, label_width, 1].
      next_image: An optional tensor of shape [image_height, image_width, 3].
      next_label: An optional tensor of shape [label_height, label_width, 1].
      depth: An optional tensor of shape [label_height, label_width, 1].
      panoptic_copy_paste_image: A tensor of shape
        [panoptic_copy_paste_image_height, panoptic_copy_paste_image_width, 3]
        or None. It is used when panoptic copy-paste is enabled.
      panoptic_copy_paste_label: A tensor of shape
        [panoptic_copy_paste_label_height, panoptic_copy_paste_label_width, 1]
        or None. It is used when panoptic copy-paste is enabled.
      frame_id: An optional string specifying the frame name.

    Returns:
      sample: A dictionary storing required data for panoptic segmentation.

    Raises:
      ValueError: An error occurs when the label shape is invalid.
      NotImplementedError: An error occurs when panoptic copy-paste comes
        together with prev_image or prev_label, not currently implemented.
      NotImplementedError: An error occurs when thing_id_mask_annotations comes
        together with prev_image or prev_label, not currently implemented.
    """
    if label is not None:
      label.get_shape().assert_is_compatible_with(
          tf.TensorShape([None, None, 1]))
      original_label = tf.cast(label, dtype=tf.int32, name='original_label')
      if next_label is not None:
        original_next_label = tf.cast(
            next_label, dtype=tf.int32, name='original_next_label')
    original_depth = None
    if depth is not None:
      # Depth label storing depth * 256 as tf.int32.
      original_depth = tf.cast(depth, dtype=tf.float32, name='original_depth')
      original_depth = original_depth / 256
    # Reusing the preprocessing function for both next and prev samples.
    if next_image is not None:
      resized_image, image, label, next_image, next_label, depth = (
          self._preprocessing_fn(
              image, label, prev_image=next_image, prev_label=next_label,
              depth=depth))
    else:
      resized_image, image, label, prev_image, prev_label, depth = (
          self._preprocessing_fn(
              image, label, prev_image=prev_image, prev_label=prev_label,
              depth=depth))

    if self._panoptic_copy_paste_options:
      if self._panoptic_copy_paste_options.panoptic_copy_paste_scale > 0.0:
        if any([
            prev_image is not None, prev_label is not None, next_image
            is not None, next_label is not None
        ]):
          raise NotImplementedError(
              'Current implementation of panoptic copy-paste does not ' +
              'support prev_image, prev_label, next_image, or next_label.')
        _, panoptic_copy_paste_image, panoptic_copy_paste_label, _, _, _ = (
            self._panoptic_copy_paste_preprocessing_fn(
                panoptic_copy_paste_image, panoptic_copy_paste_label))
        image, label = self._get_panoptic_copy_paste(
            image, label, panoptic_copy_paste_image, panoptic_copy_paste_label,
            self._panoptic_copy_paste_options.panoptic_copy_paste_thing_option,
            self._panoptic_copy_paste_options.panoptic_copy_paste_stuff_option,
            self._panoptic_copy_paste_options.panoptic_copy_paste_void_option)

    sample = {
        common.IMAGE: image
    }
    if prev_image is not None:
      sample[common.IMAGE] = tf.concat([image, prev_image], axis=2)
    if next_image is not None:
      sample[common.NEXT_IMAGE] = next_image
      sample[common.IMAGE] = tf.concat([image, next_image], axis=2)
    if label is not None:
      # Panoptic label for crowd regions will be ignore_label.
      semantic_label, panoptic_label, thing_mask, crowd_region = (
          dataset_utils.get_semantic_and_panoptic_label(
              self._dataset_info, label, self._ignore_label))
      sample[common.GT_SEMANTIC_KEY] = tf.squeeze(semantic_label, axis=2)
      semantic_weights = tf.ones_like(semantic_label, dtype=tf.float32)
      sample[common.SEMANTIC_LOSS_WEIGHT_KEY] = tf.squeeze(
          semantic_weights, axis=2)
      sample[common.GT_IS_CROWD] = tf.squeeze(crowd_region, axis=2)

      if not self._only_semantic_annotations:
        # The sample will have the original label including crowd regions.
        sample[common.GT_PANOPTIC_KEY] = tf.squeeze(label, axis=2)
        # Compute center loss for all non-crowd and non-ignore pixels.
        non_crowd_and_non_ignore_regions = tf.logical_and(
            tf.logical_not(crowd_region),
            tf.not_equal(semantic_label, self._ignore_label))
        sample[common.CENTER_LOSS_WEIGHT_KEY] = tf.squeeze(tf.cast(
            non_crowd_and_non_ignore_regions, tf.float32), axis=2)
        # Compute regression loss only for thing pixels that are not crowd.
        non_crowd_things = tf.logical_and(
            tf.logical_not(crowd_region), thing_mask)
        sample[common.REGRESSION_LOSS_WEIGHT_KEY] = tf.squeeze(tf.cast(
            non_crowd_things, tf.float32), axis=2)

        prev_panoptic_label = None
        next_panoptic_label = None
        if prev_label is not None:
          _, prev_panoptic_label, _, _ = (
              dataset_utils.get_semantic_and_panoptic_label(
                  self._dataset_info, prev_label, self._ignore_label))
        if next_label is not None:
          _, next_panoptic_label, _, _ = (
              dataset_utils.get_semantic_and_panoptic_label(
                  self._dataset_info, next_label, self._ignore_label))
        (sample[common.GT_INSTANCE_CENTER_KEY],
         sample[common.GT_INSTANCE_REGRESSION_KEY],
         sample[common.SEMANTIC_LOSS_WEIGHT_KEY],
         prev_center_map,
         frame_center_offsets,
         next_offset) = self._generate_gt_center_and_offset(
             panoptic_label, semantic_weights, prev_panoptic_label,
             next_panoptic_label)

        sample[common.GT_INSTANCE_REGRESSION_KEY] = tf.cast(
            sample[common.GT_INSTANCE_REGRESSION_KEY], tf.float32)

        if next_label is not None:
          sample[common.GT_NEXT_INSTANCE_REGRESSION_KEY] = tf.cast(
              next_offset, tf.float32)
          sample[common.NEXT_REGRESSION_LOSS_WEIGHT_KEY] = tf.cast(
              tf.greater(tf.reduce_sum(tf.abs(next_offset), axis=2), 0),
              tf.float32)

        # Only squeeze center map and semantic loss weights, as regression map
        # has two channels (x and y offsets).
        sample[common.GT_INSTANCE_CENTER_KEY] = tf.squeeze(
            sample[common.GT_INSTANCE_CENTER_KEY], axis=2)
        sample[common.SEMANTIC_LOSS_WEIGHT_KEY] = tf.squeeze(
            sample[common.SEMANTIC_LOSS_WEIGHT_KEY], axis=2)

        if prev_label is not None:
          sample[common.GT_FRAME_OFFSET_KEY] = frame_center_offsets
          sample[common.GT_FRAME_OFFSET_KEY] = tf.cast(
              sample[common.GT_FRAME_OFFSET_KEY], tf.float32)
          frame_offsets_present = tf.logical_or(
              tf.not_equal(frame_center_offsets[..., 0], 0),
              tf.not_equal(frame_center_offsets[..., 1], 0))
          sample[common.FRAME_REGRESSION_LOSS_WEIGHT_KEY] = tf.cast(
              frame_offsets_present, tf.float32)
          if self._is_training:
            sample[common.IMAGE] = tf.concat(
                [sample[common.IMAGE], prev_center_map], axis=2)

        if self._thing_id_mask_annotations:
          if any([prev_image is not None,
                  prev_label is not None,
                  next_image is not None,
                  next_label is not None]):
            raise NotImplementedError(
                'Current implementation of Max-DeepLab does not support '
                + 'prev_image, prev_label, next_image, or next_label.')
          thing_id_mask, thing_id_class = (
              self._generate_thing_id_mask_and_class(
                  panoptic_label, non_crowd_things))
          sample[common.GT_THING_ID_MASK_KEY] = tf.squeeze(
              thing_id_mask, axis=2)
          sample[common.GT_THING_ID_CLASS_KEY] = thing_id_class

    if depth is not None:
      # Depth maps are stored as an array of integers of depth * 256.
      depth = tf.cast(depth, tf.float32)
      depth = depth / 256
      sample[common.GT_DEPTH_KEY] = depth

    if not self._is_training:
      # Resized image is only used during visualization.
      sample[common.RESIZED_IMAGE] = resized_image
      sample[common.IMAGE_NAME] = image_name
      sample[common.GT_SIZE_RAW] = tf.stack([height, width], axis=0)
      if self._dataset_info['is_video_dataset']:
        sample[common.SEQUENCE_ID] = sequence
        sample[common.FRAME_ID] = frame_id
      # Keep original labels for evaluation.
      if label is not None:
        orig_semantic_label, _, _, orig_crowd_region = (
            dataset_utils.get_semantic_and_panoptic_label(
                self._dataset_info, original_label, self._ignore_label))
        sample[common.GT_SEMANTIC_RAW] = tf.squeeze(orig_semantic_label, axis=2)
        if not self._only_semantic_annotations:
          sample[common.GT_PANOPTIC_RAW] = tf.squeeze(original_label, axis=2)
          sample[common.GT_IS_CROWD_RAW] = tf.squeeze(orig_crowd_region)
          if next_label is not None:
            sample[common.GT_NEXT_PANOPTIC_RAW] = tf.squeeze(
                original_next_label, axis=2)
      if depth is not None:
        sample[common.GT_DEPTH_RAW] = original_depth

    return sample

  def _generate_thing_id_mask_and_class(self,
                                        panoptic_label,
                                        non_crowd_things):
    """Generates the ground-truth thing-ID masks and their class labels.

    It computes thing-ID mask and class with unique ID for each thing instance.
    `thing_id` indicates the number of unique thing-ID to each instance in an
    image, starting the counting from 0. Each pixel in thing_id_mask is labeled
    with the corresponding thing-ID.

    Args:
      panoptic_label: A tf.Tensor of shape [height, width, 1].
      non_crowd_things: A tf.Tensor of shape [height, width, 1], indicating
        non-crowd and thing-class regions.

    Returns:
      thing_id_mask: A tf.Tensor of shape [height, width, 1]. It assigns each
        non-crowd thing instance a unique mask-ID label, starting from 0.
        Unassigned pixels are set to -1.
      thing_id_class: A tf.Tensor of shape [max_thing_id]. It contains semantic
        ID of each instance assigned to thing_id_mask. The remaining
        (max_thing_id - num_things) elements are set to -1.

    Raises:
      ValueError: An error occurs when the thing-ID mask contains stuff or crowd
        region.
      ValueError: An error occurs when thing_count is greater or equal to
        self._max_thing_id.

    """
    unique_ids, _ = tf.unique(tf.reshape(panoptic_label, [-1]))
    thing_id_mask = -tf.ones_like(panoptic_label)
    thing_id_class = -tf.ones(self._max_thing_id)
    thing_count = 0
    for panoptic_id in unique_ids:
      semantic_id = panoptic_id // self._dataset_info['panoptic_label_divisor']
      # Filter out IDs that are not thing instances (i.e., IDs for ignore_label,
      # stuff classes or crowd). Stuff classes and crowd regions both have IDs
      # of the form panoptic_id = semantic_id * label_divisor (i.e., instance id
      # = 0)
      if (semantic_id == self._dataset_info['ignore_label'] or
          panoptic_id % self._dataset_info['panoptic_label_divisor'] == 0):
        continue

      assert_stuff_crowd = tf.debugging.Assert(
          tf.reduce_all(non_crowd_things[panoptic_label == panoptic_id]),
          ['thing-ID mask here must not contain stuff or crowd region.'])
      with tf.control_dependencies([assert_stuff_crowd]):
        panoptic_id = tf.identity(panoptic_id)

      thing_id_mask = tf.where(panoptic_label == panoptic_id,
                               thing_count, thing_id_mask)

      assert_thing_count = tf.debugging.Assert(
          thing_count < self._max_thing_id,
          ['thing_count must be smaller than self._max_thing_id.'])
      with tf.control_dependencies([assert_thing_count]):
        thing_count = tf.identity(thing_count)

      thing_id_class = tf.tensor_scatter_nd_update(
          thing_id_class, [[thing_count]], [semantic_id])
      thing_count += 1
    return thing_id_mask, thing_id_class

  def _generate_prev_centers_with_noise(self,
                                        panoptic_label,
                                        offset_noise_factor=0.05,
                                        false_positive_rate=0.2,
                                        false_positive_noise_factor=0.05):
    """Generates noisy center predictions for the previous frame.

    Args:
      panoptic_label: A tf.Tensor of shape [height, width, 1].
      offset_noise_factor: An optional float defining the maximum fraction of
        the object size that is used to displace the previous center.
      false_positive_rate: An optional float indicating at which probability
        false positives should be added.
      false_positive_noise_factor: An optional float defining the maximum
        fraction of the object size that is used to displace the false positive
        center.

    Returns:
      A tuple of (center, ids_to_center) with both being tf.Tensor of shape
      [height, width, 1] and shape [N, 2] where N is the number of unique IDs.
    """
    height = tf.shape(panoptic_label)[0]
    width = tf.shape(panoptic_label)[1]

    # Pad center to make boundary handling easier.
    center_pad_begin = int(round(3 * self._sigma + 1))
    center_pad_end = int(round(3 * self._sigma + 2))
    center_pad = center_pad_begin + center_pad_end

    center = tf.zeros((height + center_pad, width + center_pad))
    unique_ids, _ = tf.unique(tf.reshape(panoptic_label, [-1]))
    ids_to_center_x = tf.zeros_like(unique_ids, dtype=tf.int32)
    ids_to_center_y = tf.zeros_like(unique_ids, dtype=tf.int32)

    for panoptic_id in unique_ids:
      semantic_id = panoptic_id // self._dataset_info['panoptic_label_divisor']
      # Filter out IDs that should be ignored, are stuff classes or crowd.
      # Stuff classes and crowd regions both have IDs of the form panoptic_id =
      # semantic_id * label_divisor
      if (semantic_id == self._dataset_info['ignore_label'] or
          panoptic_id % self._dataset_info['panoptic_label_divisor'] == 0):
        continue

      # Convert [[y0, x0, 0], ...] to [[y0, ...], [x0, ...], [0, ...]].
      mask_index = tf.cast(
          tf.transpose(tf.where(panoptic_label == panoptic_id)), tf.float32)
      centers = tf.reduce_mean(mask_index, axis=1)
      bbox_size = (
          tf.reduce_max(mask_index, axis=1) - tf.reduce_min(mask_index, axis=1))

      # Add noise.
      center_y = (
          centers[0] + tf.random.normal([], dtype=tf.float32) *
          offset_noise_factor * bbox_size[0])
      center_x = (
          centers[1] + tf.random.normal([], dtype=tf.float32) *
          offset_noise_factor * bbox_size[1])

      center_x = tf.minimum(
          tf.maximum(tf.cast(tf.round(center_x), tf.int32), 0), width - 1)
      center_y = tf.minimum(
          tf.maximum(tf.cast(tf.round(center_y), tf.int32), 0), height - 1)

      id_index = tf.where(tf.equal(panoptic_id, unique_ids))
      ids_to_center_x = tf.tensor_scatter_nd_update(
          ids_to_center_x, id_index, tf.expand_dims(center_x, axis=0))
      ids_to_center_y = tf.tensor_scatter_nd_update(
          ids_to_center_y, id_index, tf.expand_dims(center_y, axis=0))

      def add_center_gaussian(center_x_coord, center_y_coord, center):
        # Due to the padding with center_pad_begin in center, the computed
        # center becomes the upper left corner in the center tensor.
        upper_left = center_x_coord, center_y_coord
        bottom_right = (upper_left[0] + self._gaussian_size,
                        upper_left[1] + self._gaussian_size)

        indices_x, indices_y = tf.meshgrid(
            tf.range(upper_left[0], bottom_right[0]),
            tf.range(upper_left[1], bottom_right[1]))
        indices = tf.transpose(
            tf.stack([tf.reshape(indices_y, [-1]),
                      tf.reshape(indices_x, [-1])]))

        return tf.tensor_scatter_nd_max(
            center, indices, self._gaussian, name='center_scatter')

      center = add_center_gaussian(center_x, center_y, center)
      # Generate false positives.
      center_y = (
          tf.cast(center_y, dtype=tf.float32) +
          tf.random.normal([], dtype=tf.float32) * false_positive_noise_factor *
          bbox_size[0])
      center_x = (
          tf.cast(center_x, dtype=tf.float32) +
          tf.random.normal([], dtype=tf.float32) * false_positive_noise_factor *
          bbox_size[1])

      center_x = tf.minimum(
          tf.maximum(tf.cast(tf.round(center_x), tf.int32), 0), width - 1)
      center_y = tf.minimum(
          tf.maximum(tf.cast(tf.round(center_y), tf.int32), 0), height - 1)
      # Draw a sample to decide whether to add a false positive or not.
      center = center + tf.cast(
          tf.random.uniform([], dtype=tf.float32) < false_positive_rate,
          tf.float32) * (
              add_center_gaussian(center_x, center_y, center) - center)

    center = center[center_pad_begin:(center_pad_begin + height),
                    center_pad_begin:(center_pad_begin + width)]
    center = tf.expand_dims(center, -1)
    return center, unique_ids, ids_to_center_x, ids_to_center_y

  def _generate_gt_center_and_offset(self,
                                     panoptic_label,
                                     semantic_weights,
                                     prev_panoptic_label=None,
                                     next_panoptic_label=None):
    """Generates the ground-truth center and offset from the panoptic labels.

    Additionally, the per-pixel weights for the semantic branch are increased
    for small instances. In case, prev_panoptic_label is passed, it also
    computes the previous center heatmap with random noise and the offsets
    between center maps.

    Args:
      panoptic_label: A tf.Tensor of shape [height, width, 1].
      semantic_weights: A tf.Tensor of shape [height, width, 1].
      prev_panoptic_label: An optional tf.Tensor of shape [height, width, 1].
      next_panoptic_label: An optional tf.Tensor of shape [height, width, 1].

    Returns:
      A tuple (center, offsets, weights, prev_center, frame_offset*,
      next_offset) with each being a tf.Tensor of shape [height, width, 1 (2*)].
      If prev_panoptic_label is None, prev_center and frame_offset are None.
      If next_panoptic_label is None, next_offset is None.
    """
    height = tf.shape(panoptic_label)[0]
    width = tf.shape(panoptic_label)[1]

    # Pad center to make boundary handling easier.
    center_pad_begin = int(round(3 * self._sigma + 1))
    center_pad_end = int(round(3 * self._sigma + 2))
    center_pad = center_pad_begin + center_pad_end

    center = tf.zeros((height + center_pad, width + center_pad))
    offset_x = tf.zeros((height, width, 1), dtype=tf.int32)
    offset_y = tf.zeros((height, width, 1), dtype=tf.int32)
    unique_ids, _ = tf.unique(tf.reshape(panoptic_label, [-1]))

    prev_center = None
    frame_offsets = None
    # Due to loop handling in tensorflow, these variables had to be defined for
    # all cases.
    frame_offset_x = tf.zeros((height, width, 1), dtype=tf.int32)
    frame_offset_y = tf.zeros((height, width, 1), dtype=tf.int32)

    # Next-frame instance offsets.
    next_offset = None
    next_offset_y = tf.zeros((height, width, 1), dtype=tf.int32)
    next_offset_x = tf.zeros((height, width, 1), dtype=tf.int32)

    if prev_panoptic_label is not None:
      (prev_center, prev_unique_ids, prev_centers_x, prev_centers_y
      ) = self._generate_prev_centers_with_noise(prev_panoptic_label)

    for panoptic_id in unique_ids:
      semantic_id = panoptic_id // self._dataset_info['panoptic_label_divisor']
      # Filter out IDs that should be ignored, are stuff classes or crowd.
      # Stuff classes and crowd regions both have IDs of the form panopti_id =
      # semantic_id * label_divisor
      if (semantic_id == self._dataset_info['ignore_label'] or
          panoptic_id % self._dataset_info['panoptic_label_divisor'] == 0):
        continue

      # Convert [[y0, x0, 0], ...] to [[y0, ...], [x0, ...], [0, ...]].
      mask_index = tf.transpose(tf.where(panoptic_label == panoptic_id))
      mask_y_index = mask_index[0]
      mask_x_index = mask_index[1]

      next_mask_index = None
      next_mask_y_index = None
      next_mask_x_index = None
      if next_panoptic_label is not None:
        next_mask_index = tf.transpose(
            tf.where(next_panoptic_label == panoptic_id))
        next_mask_y_index = next_mask_index[0]
        next_mask_x_index = next_mask_index[1]

      instance_area = tf.shape(mask_x_index)
      if instance_area < self._instance_area_threshold:
        semantic_weights = tf.where(panoptic_label == panoptic_id,
                                    self._small_instance_weight,
                                    semantic_weights)

      centers = tf.reduce_mean(tf.cast(mask_index, tf.float32), axis=1)

      center_x = tf.cast(tf.round(centers[1]), tf.int32)
      center_y = tf.cast(tf.round(centers[0]), tf.int32)

      # Due to the padding with center_pad_begin in center, the computed center
      # becomes the upper left corner in the center tensor.
      upper_left = center_x, center_y
      bottom_right = (upper_left[0] + self._gaussian_size,
                      upper_left[1] + self._gaussian_size)

      indices_x, indices_y = tf.meshgrid(
          tf.range(upper_left[0], bottom_right[0]),
          tf.range(upper_left[1], bottom_right[1]))
      indices = tf.transpose(
          tf.stack([tf.reshape(indices_y, [-1]),
                    tf.reshape(indices_x, [-1])]))

      center = tf.tensor_scatter_nd_max(
          center, indices, self._gaussian, name='center_scatter')
      offset_y = tf.tensor_scatter_nd_update(
          offset_y,
          tf.transpose(mask_index),
          center_y - tf.cast(mask_y_index, tf.int32),
          name='offset_y_scatter')
      offset_x = tf.tensor_scatter_nd_update(
          offset_x,
          tf.transpose(mask_index),
          center_x - tf.cast(mask_x_index, tf.int32),
          name='offset_x_scatter')
      if prev_panoptic_label is not None:
        mask = tf.equal(prev_unique_ids, panoptic_id)
        if tf.math.count_nonzero(mask) > 0:
          prev_center_x = prev_centers_x[mask]
          prev_center_y = prev_centers_y[mask]

          frame_offset_y = tf.tensor_scatter_nd_update(
              frame_offset_y,
              tf.transpose(mask_index),
              prev_center_y - tf.cast(mask_y_index, tf.int32),
              name='frame_offset_y_scatter')
          frame_offset_x = tf.tensor_scatter_nd_update(
              frame_offset_x,
              tf.transpose(mask_index),
              prev_center_x - tf.cast(mask_x_index, tf.int32),
              name='frame_offset_x_scatter')
      if next_panoptic_label is not None:
        next_offset_y = tf.tensor_scatter_nd_update(
            next_offset_y,
            tf.transpose(next_mask_index),
            center_y - tf.cast(next_mask_y_index, tf.int32),
            name='next_offset_y_scatter')
        next_offset_x = tf.tensor_scatter_nd_update(
            next_offset_x,
            tf.transpose(next_mask_index),
            center_x - tf.cast(next_mask_x_index, tf.int32),
            name='next_offset_x_scatter')

    offset = tf.concat([offset_y, offset_x], axis=2)
    center = center[center_pad_begin:(center_pad_begin + height),
                    center_pad_begin:(center_pad_begin + width)]
    center = tf.expand_dims(center, -1)
    if prev_panoptic_label is not None:
      frame_offsets = tf.concat([frame_offset_y, frame_offset_x], axis=2)
    if next_panoptic_label is not None:
      next_offset = tf.concat([next_offset_y, next_offset_x], axis=2)
    return (center, offset, semantic_weights, prev_center, frame_offsets,
            next_offset)

  def _get_panoptic_copy_paste(self, image, label, panoptic_copy_paste_image,
                               panoptic_copy_paste_label,
                               panoptic_copy_paste_thing_option,
                               panoptic_copy_paste_stuff_option,
                               panoptic_copy_paste_void_option):
    """Applies Panoptic Copy-Paste augmentation given two images and labels.

    The panoptic copy-paste augmentation will randomly copy-and-paste an image
    onto another one. A prerequest is that maximum of instance number of any
    sample should not exceed label_divisor // 2, otherwise the total number of
    instance can exceed label_divisor and cause errors.

    The panoptic copy-paste [1] is an extension of thing-specific copy-paste
    augmentation [2,3] to panoptic segmentation task.

    References:
      [1] k-means Mask Transformer, ECCV 2022.
            Qihang Yu, Huiyu Wang, Siyuan Qiao, Maxwell Collins, Yukun Zhu,
            Hartwig Adam, Alan Yuille, Liang-Chieh Chen.

      [2] Instaboost: Boosting instance segmentation via probability map
          guided copy-pasting, ICCV 2019
            Hao-Shu Fang, Jianhua Sun, Runzhong Wang, Minghao Gou, Yong-Lu Li,
            Cewu Lu.

      [3] Simple copy-paste is a strong data augmentation method for
          instance segmentation, CVPR 2021.
            Golnaz Ghiasi, Yin Cui, Aravind Srinivas, Rui Qian, Tsung-Yi Lin,
            Ekin D. Cubuk, Quoc V. Le, Barret Zoph.

    Args:
      image: A tf.Tensor of shape [image_height, image_width, 3],
        the image as canvas.
      label: A tf.Tensor of shape [label_height, label_width, 1].
      panoptic_copy_paste_image: A tf.Tensor of shape
        [panoptic_copy_paste_image_height, panoptic_copy_paste_image_width, 3],
        the image to be copyed and pasted.
      panoptic_copy_paste_label: A tensor of shape
        [panoptic_copy_paste_label_height, panoptic_copy_paste_label_width, 1].
      panoptic_copy_paste_thing_option: A str indicating how to deal with thing
        class.
      panoptic_copy_paste_stuff_option: A str indicating how to deal with stuff
        class.
      panoptic_copy_paste_void_option: A str indicating how to deal with void
        class.

    Returns:
      A tf.Tensor of the image that has been panoptic copy-paste augmented, and
        a tf.Tensor of the updated label.

    Raises:
      tf.errors.InvalidArgumentError: An error occurs if maximum instance id
        exceeds label_divisor // 2.
    """
    label_divisor = self._dataset_info['panoptic_label_divisor']
    semantic_ignore_label = self._ignore_label
    panoptic_ignore_label = semantic_ignore_label * label_divisor

    # The values indicating pixels that are padded, void, thing, and stuff,
    # respectively.
    padding_list = tf.constant([self._panoptic_copy_paste_padding_value],
                               tf.int32)
    void_list = tf.constant([panoptic_ignore_label], tf.int32)
    thing_list = list(self._dataset_info['class_has_instances_list'])
    num_thing_stuff_classes = utils.get_num_thing_stuff_classes(
        self._dataset_info['num_classes'], semantic_ignore_label)
    stuff_list = utils.get_stuff_class_ids(num_thing_stuff_classes, thing_list,
                                           semantic_ignore_label)

    # Flatten and get unique label id in panoptic_copy_paste_label.
    unique_label_id, unique_label_idx_per_pixel = tf.unique(
        tf.reshape(panoptic_copy_paste_label, [-1]))
    # Pick a random number of unique labels that should be kept.
    number_of_unique_label_id_to_keep = tf.random.uniform(
        shape=[], maxval=tf.shape(unique_label_id)[0] + 1, dtype=tf.int32)
    unique_label_id_mask = tf.concat([
        tf.ones([number_of_unique_label_id_to_keep], tf.int32),
        tf.zeros(
            [tf.shape(unique_label_id)[0] - number_of_unique_label_id_to_keep],
            tf.int32)], axis=0)
    # shuffled_unique_x_mask is one for pixels that should be kept, and 0
    # otherwise.
    shuffled_unique_label_id_mask = tf.random.shuffle(unique_label_id_mask)

    def whether_in_list(tensor, value_list):
      """Whether each value in tensor exists in value_list.

      For each value in tensor, whether it exists in value_list, if so,
      return 1, otherwise return 0.

      Args:
        tensor: A tf.Tensor with shape [N].
        value_list: A python list with length M or tf.Tensor with shape [M].

      Returns:
        A tf.Tensor of shape [N] with value 1 of the element exists in
          value_list, otherwise 0.
      """
      expand_tensor = tf.expand_dims(tensor, -1)
      expand_value_list = tf.expand_dims(value_list, 0)
      broadcast_equal = tf.equal(expand_tensor, expand_value_list)
      broadcast_equal_int = tf.cast(broadcast_equal, tf.int32)
      return tf.reduce_max(broadcast_equal_int, axis=-1)

    padding_list_mask = whether_in_list(unique_label_id, padding_list)
    void_list_mask = whether_in_list(unique_label_id, void_list)
    thing_list_mask = whether_in_list(unique_label_id // label_divisor,
                                      thing_list)
    stuff_list_mask = whether_in_list(unique_label_id // label_divisor,
                                      stuff_list)

    def update_label_id_mask(label_id_mask, class_list_mask, option):
      """Updates the label_id to be kept based on semantic class and option.

      Args:
        label_id_mask: A tf.Tensor of shape [N], with value 0 for discarded
          semantic class and 1 otherwise.
        class_list_mask: A tf.Tensor of shape [N], with value 0 for semantic
          class that does not belong to some specific class_list (e.g.,
          thing_class_list), and 1 otherwise.
        option: A str, one of ['none', 'random', 'all']:
          'none': All semantic classes in the class_list will be discarded based
            on class_list_mask.
          'random': No further change on label_id_mask, thus the kept semantic
            class is randomly choosed.
          'all': All semantic class in the class_list will be kept based on
            class_list_mask.

      Returns:
        An updated tf.Tensor of shape [N], with value 0 for discarded
          semantic class and 1 otherwise.
      """
      # Drop all pixels with class in class_list.
      if option == 'none':
        return label_id_mask * (1 - class_list_mask)
      elif option == 'random':
        return label_id_mask
      # Keep all pixels with class in class_list.
      elif option == 'all':
        return tf.maximum(label_id_mask, class_list_mask)

    # Drop padded pixels.
    shuffled_unique_label_id_mask = update_label_id_mask(
        shuffled_unique_label_id_mask, padding_list_mask, option='none')

    shuffled_unique_label_id_mask = update_label_id_mask(
        shuffled_unique_label_id_mask, thing_list_mask,
        panoptic_copy_paste_thing_option)
    shuffled_unique_label_id_mask = update_label_id_mask(
        shuffled_unique_label_id_mask, stuff_list_mask,
        panoptic_copy_paste_stuff_option)
    shuffled_unique_label_id_mask = update_label_id_mask(
        shuffled_unique_label_id_mask, void_list_mask,
        panoptic_copy_paste_void_option)

    # Gather from unique labels to per pixel mask.
    per_pixel_mask = tf.gather(shuffled_unique_label_id_mask,
                               unique_label_idx_per_pixel)
    panoptic_copy_paste_ignore_pixel = 1 - tf.reshape(
        per_pixel_mask, tf.shape(panoptic_copy_paste_label))

    panoptic_copy_paste_ignore_pixel_int = tf.cast(
        panoptic_copy_paste_ignore_pixel, tf.int32)
    panoptic_copy_paste_ignore_pixel_float = tf.cast(
        panoptic_copy_paste_ignore_pixel, tf.float32)
    # Copy-paste panoptic_copy_paste_image onto image based on
    # panoptic_copy_paste_ignore_pixel_float.
    image = (
        image * panoptic_copy_paste_ignore_pixel_float +
        panoptic_copy_paste_image *
        (1.0 - panoptic_copy_paste_ignore_pixel_float))

    semantic_label = label // label_divisor
    instance_label = label % label_divisor
    panoptic_label = semantic_label * label_divisor + instance_label

    panoptic_copy_paste_semantic_label = (
        panoptic_copy_paste_label // label_divisor)
    panoptic_copy_paste_instance_label = (
        panoptic_copy_paste_label % label_divisor)
    max_instance_assert = tf.Assert(
        tf.logical_and(
            tf.less(tf.reduce_max(instance_label), label_divisor // 2),
            tf.less(
                tf.reduce_max(panoptic_copy_paste_instance_label),
                label_divisor // 2)),
        ['Max instance number shall not exceeds half of label_divisor when' +
         ' panoptic copy-paste augmentation is enabled!'])
    with tf.control_dependencies([max_instance_assert]):
      # Invert the instance id from increasing from 1 to decreasing from
      # label_divisor, but keep instance_id = 0 for stuff/void regions.
      panoptic_copy_paste_instance_label = tf.where(
          panoptic_copy_paste_instance_label == 0,
          panoptic_copy_paste_instance_label,
          label_divisor - 1 - panoptic_copy_paste_instance_label)
      panoptic_copy_paste_panoptic_label = (
          panoptic_copy_paste_semantic_label * label_divisor +
          panoptic_copy_paste_instance_label)

      label = (
          panoptic_label * panoptic_copy_paste_ignore_pixel_int +
          panoptic_copy_paste_panoptic_label *
          (1 - panoptic_copy_paste_ignore_pixel_int))

      return image, label


class MultiCameraPanopticSampleGenerator(object):
  """This class generates samples from multi-camera images and labels."""

  def __init__(self, *args, **kwargs):
    self._view_sample_generator = PanopticSampleGenerator(*args, **kwargs)

  def __call__(self, sample_dict, panoptic_copy_psate_sample_dict=None):
    if panoptic_copy_psate_sample_dict is not None:
      raise NotImplementedError(
          'MultiCamera is incompatible with panoptic_copy_paste.')
    camera_datas = {}
    for camera_name, val in sample_dict.items():
      val = self._view_sample_generator(val)
      camera_datas[camera_name] = val
    return camera_datas
