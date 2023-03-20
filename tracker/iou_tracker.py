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

r"""This file contains code to track based on IoU overlaps.

The IoUTracker takes frame-by-frame panoptic segmentation prediction and
generates video panoptic segmentation with re-ordered identities based on IoU
overlaps within consecutive frames.

We recommend to use the 3 input channels as it allows for a wide range of
instance IDs. The evaluator options can be configured with raw_panoptic_format:
`three_channel_png` to export results in the recommended format.

To run this script, you need to install scipy.
For example, install it via pip:
$pip install scipy
"""

import collections
import os
import pprint
from typing import List, Text, Tuple, Optional

from absl import app
from absl import flags
from absl import logging
import numpy as np
from scipy import optimize
import tensorflow as tf

from deeplab2.data import dataset
from deeplab2.evaluation import segmentation_and_tracking_quality as stq
from deeplab2.tracker import optical_flow_utils
from deeplab2.trainer import vis_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'gt', None, 'The path to the gt video frames. This folder '
    'should contain one folder per sequence.')
flags.DEFINE_string(
    'pred', None, 'The path to the prediction video frames. '
    'This folder should contain one folder per sequence.')
flags.DEFINE_string(
    'output', '', 'The path to store the tracked video frames.'
    'This folder should contain one folder per sequence.')
flags.DEFINE_string('sequence', '', 'The sequence ID to evaluate on.')
flags.DEFINE_string(
    'dataset', 'kitti_step', 'The specified dataset is used'
    ' to interpret the labels. Supported options are: ' +
    ', '.join(dataset.MAP_NAMES))
flags.DEFINE_string(
    'optical_flow', None,
    'The path to the optical flow predictions. This folder '
    'should contain one folder per sequence.')
flags.DEFINE_integer(
    'input_channels', 2, 'DeepLab2 supports two formats when exporting '
    'predictions. The first channel of the input always encodes the semantic '
    'class while either only the second channel (G in RGB) encodes the '
    'instance ID or the second and third channel (GB in RGB). Depending on the '
    'ground-truth and prediction format, the valid options are `2` and `3`.')

_LABEL_DIVISOR = 10000
_OCCLUSION_EXT = '.occ_forward'
_FLOW_EXT = '.flow_forward'


def _format_output(output, indent=4):
  """Formats `output`, either on one line, or indented across multiple lines."""
  formatted = pprint.pformat(output)
  lines = formatted.splitlines()
  if len(lines) == 1:
    return formatted
  lines = [' ' * indent + line for line in lines]
  return '\n' + '\n'.join(lines)


def _compute_mask_iou(instance_a: np.ndarray, instance_b: np.ndarray) -> int:
  """Computes the IoU of two binary masks."""
  intersection = np.count_nonzero(
      np.logical_and(instance_a > 0, instance_b > 0).astype(np.uint8))
  non_intersection_a = np.count_nonzero(instance_a > 0) - intersection
  non_intersection_b = np.count_nonzero(instance_b > 0) - intersection
  return intersection / (intersection + non_intersection_a + non_intersection_b)


class IoUTracker(object):
  """This class computes track IDs based on IoU overlap."""

  def __init__(self,
               classes_to_track: List[int],
               label_divisor: int,
               sigma=10,
               iou_threshold=0.3):
    """Initializes the tracker.

    Args:
      classes_to_track: A list of class IDs that should be tracked.
      label_divisor: The divisor to split the label map into semantic classes
        and instance IDs.
      sigma: An integer specifying the number of frames that tracks should be
        kept active while being discontinued.
      iou_threshold: A float specifying the minimum IoU value for a match.
    """
    self._sigma = sigma
    self._iou_threshold = iou_threshold
    self._classes_to_track = classes_to_track
    self._label_divisor = label_divisor
    self.reset_states()

  def reset_states(self):
    """Resets all tracking states."""
    self._last_mask_per_track = {
        i: collections.OrderedDict() for i in self._classes_to_track
    }
    self._frames_since_last_update = {
        i: collections.OrderedDict() for i in self._classes_to_track
    }
    # `0` is reserved for `crowd`.
    self._next_track_id = 1

  def _add_track(self, object_mask: np.ndarray, class_index: int):
    """Adds a new track."""
    track_id = self._next_track_id
    self._last_mask_per_track[class_index][track_id] = object_mask
    self._frames_since_last_update[class_index][track_id] = 0
    self._next_track_id += 1

  def _remove_track(self, track_id: int, class_index: int):
    """Removes a track."""
    del self._last_mask_per_track[class_index][track_id]
    del self._frames_since_last_update[class_index][track_id]

  def _increase_inactivity_of_track(self, track_id: int, class_index: int):
    """Increases inactivity of track and potentially remove it."""
    self._frames_since_last_update[class_index][track_id] += 1
    if self._frames_since_last_update[class_index][track_id] > self._sigma:
      self._remove_track(track_id, class_index)

  def _match_instances_to_tracks(
      self, instances: List[np.ndarray], class_index: int,
      instances_with_track_id: np.ndarray,
      warped_instances: List[np.ndarray]) -> np.ndarray:
    """Match instances to tracks and update tracks accordingly."""
    track_ids = list(self._last_mask_per_track[class_index].keys())

    # Match instances to tracks based on IoU overlap.
    if warped_instances:
      matches, unmatched_instances, unmatched_tracks = (
          self._associate_instances_to_tracks(warped_instances, class_index))
    else:
      matches, unmatched_instances, unmatched_tracks = (
          self._associate_instances_to_tracks(instances, class_index))

    # Extend existing tracks.
    for instance_index, track_id_index in matches:
      track_id = track_ids[track_id_index]
      instance_mask = instances[instance_index]
      self._last_mask_per_track[class_index][track_id] = instance_mask
      self._frames_since_last_update[class_index][track_id] = 0
      instances_with_track_id[instance_mask] = track_id

    # Add new tracks.
    for instance_index in unmatched_instances:
      instance_mask = instances[instance_index]
      self._add_track(instance_mask, class_index)
      instances_with_track_id[instance_mask] = self._next_track_id - 1

    # Remove tracks that are inactive for more than `sigma` frames.
    for track_id_index in unmatched_tracks:
      track_id = track_ids[track_id_index]
      self._increase_inactivity_of_track(track_id, class_index)

    return instances_with_track_id

  def update(self, predicted_frame: np.ndarray,
             predicted_flow: Optional[np.ndarray],
             predicted_occlusion: Optional[np.ndarray]) -> np.ndarray:
    """Updates the tracking states and computes the track IDs.

    Args:
      predicted_frame: The panoptic label map for a particular video frame.
      predicted_flow: An optional np.array containing the optical flow.
      predicted_occlusion: An optional np.array containing the predicted
        occlusion map.

    Returns:
      The updated panoptic label map for the input frame containing track IDs.
    """
    predicted_classes = predicted_frame // self._label_divisor
    predicted_instances = predicted_frame % self._label_divisor

    instances_with_track_id = np.zeros_like(predicted_instances)

    for class_index in self._classes_to_track:
      instances_mask = np.logical_and(predicted_classes == class_index,
                                      predicted_instances > 0)
      instance_ids = np.unique(predicted_instances[instances_mask])
      instances = [
          np.logical_and(instances_mask, predicted_instances == i)
          for i in instance_ids
      ]
      # If current class has no instances, check if tracks needs to be removed,
      # because they are inactive for more than `sigma` frames.
      if not instances:
        immutable_key_list = list(self._frames_since_last_update[class_index])
        for track_id in immutable_key_list:
          self._increase_inactivity_of_track(track_id, class_index)
        continue

      # If there are no tracks recorded yet, all all instances as new tracks.
      if not self._last_mask_per_track[class_index]:
        for instance_mask in instances:
          self._add_track(instance_mask, class_index)
          instances_with_track_id[instance_mask] = self._next_track_id - 1
      else:
        # If optical flow is used, warp all instances.
        warped_instances = []
        if predicted_occlusion is not None and predicted_flow is not None:
          for instance in instances:
            warped_instance = optical_flow_utils.warp_flow(
                instance.astype(np.float32), predicted_flow)
            warped_instances.append(
                optical_flow_utils.remove_occlusions(warped_instance,
                                                     predicted_occlusion))
        instances_with_track_id = self._match_instances_to_tracks(
            instances, class_index, instances_with_track_id, warped_instances)

    if self._next_track_id >= self._label_divisor:
      raise ValueError('To many tracks were detected for the given '
                       'label_divisor. Please increase the label_divisor to '
                       'make sure that the track Ids are less than the '
                       'label_divisor.')

    return predicted_classes * self._label_divisor + instances_with_track_id

  def _associate_instances_to_tracks(
      self, instances: List[np.ndarray],
      class_index: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Matches the instances to existing tracks.

    Args:
      instances: A list of numpy arrays specifying the instance masks.
      class_index: An integer specifying the class index.

    Returns:
      A tuple of Lists:
      - Containing all indices of matches between instances and tracks.
      - Containing all indices of unmatched instances.
      - Containing all indices of unmatched tracks.
    """
    number_of_instances = len(instances)
    number_of_tracks = len(self._last_mask_per_track[class_index])
    iou_matrix = np.zeros((number_of_instances, number_of_tracks))

    for i, instance_mask in enumerate(instances):
      for j, last_mask in enumerate(
          self._last_mask_per_track[class_index].values()):
        iou_matrix[i, j] = _compute_mask_iou(instance_mask, last_mask)

    matches_indices = np.stack(
        list(optimize.linear_sum_assignment(-iou_matrix)), axis=1)
    unmatched_instances = [
        inst_id for inst_id in range(number_of_instances)
        if inst_id not in matches_indices[:, 0]
    ]
    unmatched_tracks = [
        inst_id for inst_id in range(number_of_tracks)
        if inst_id not in matches_indices[:, 1]
    ]

    list_of_matches = []
    for m in matches_indices:
      if iou_matrix[m[0], m[1]] > self._iou_threshold:
        list_of_matches.append(m)
      else:
        unmatched_instances.append(m[0])
        unmatched_tracks.append(m[1])

    return list_of_matches, unmatched_instances, unmatched_tracks


def read_panoptic_image_2ch(path: Text, label_divisor: int) -> np.ndarray:
  """Reads in a panoptic image in 2 channel format.

  The 2 channel format encodes the semantic class in the first channel, and the
  instance ID in the second channel.

  Args:
    path: A string specifying the path to the image to be loaded.
    label_divisor: An integer specifying the label divisor that is used to
      combine the semantic class and the instance ID.

  Returns:
    A numpy array enconding the semantic class and instance ID for every pixel.
  """
  with tf.io.gfile.GFile(path, 'rb') as f:
    image = tf.cast(tf.io.decode_image(f.read()), tf.int32).numpy()

  if image.shape[2] == 3 and np.any(image[..., 2] != 0):
    raise ValueError('The input %s is not in 2 channel format.' % path)
  return image[..., 0] * label_divisor + image[..., 1]


def read_panoptic_image_3ch(path: Text, label_divisor: int) -> np.ndarray:
  """Reads in a panoptic image in 3 channel format.

  The 3 channel format encodes the semantic class in the first channel, and the
  instance ID in the second and third channel as follows: instance_id =
  image[..., 1] * 256 + image[..., 2].

  Args:
    path: A string specifying the path to the image to be loaded.
    label_divisor: An integer specifying the label divisor that is used to
      combine the semantic class and the instance ID.

  Returns:
    A numpy array enconding the semantic class and instance ID for every pixel.
  """
  with tf.io.gfile.GFile(path, 'rb') as f:
    image = tf.cast(tf.io.decode_image(f.read()), tf.int32).numpy()

  return image[..., 0] * label_divisor + image[..., 1] * 256 + image[..., 2]


def read_numpy_tensor(path: Text) -> np.ndarray:
  """Reads a numpy array from `path` and returns it."""
  with tf.io.gfile.GFile(path, 'rb') as f:
    return np.load(f)


def main(unused_args):
  if FLAGS.dataset not in dataset.MAP_NAME_TO_DATASET_INFO:
    raise ValueError('Given dataset option is not a valid dataset. Please use '
                     '--help to see available options.')
  dataset_info = dataset.MAP_NAME_TO_DATASET_INFO[FLAGS.dataset]
  thing_classes = dataset_info.class_has_instances_list
  ignore_label = dataset_info.ignore_label
  num_classes = dataset_info.num_classes
  colormap_name = dataset_info.colormap
  use_optical_flow = FLAGS.optical_flow is not None

  # Create Tracker and metric.
  tracker = IoUTracker(thing_classes, _LABEL_DIVISOR)
  metric = stq.STQuality(num_classes, thing_classes, ignore_label,
                         _LABEL_DIVISOR, 256 * 256 * 256)

  if FLAGS.input_channels == 2:
    reader_fn = read_panoptic_image_2ch
  elif FLAGS.input_channels == 3:
    reader_fn = read_panoptic_image_3ch
  else:
    raise ValueError('The --input_channels must be 2 or 3.')

  # Get ground-truth files.
  for gt_sequence_folder in tf.io.gfile.glob(os.path.join(FLAGS.gt, '*')):
    tracker.reset_states()
    color_map = dict()

    sequence = os.path.basename(gt_sequence_folder)
    if FLAGS.sequence and FLAGS.sequence != sequence:
      continue
    pred_sequence_folder = os.path.join(FLAGS.pred, sequence)
    if use_optical_flow:
      optical_flow_sequence_folder = os.path.join(FLAGS.optical_flow, sequence)

    for gt_frame_path in sorted(
        tf.io.gfile.glob(os.path.join(gt_sequence_folder, '*.png'))):
      gt_frame_name = gt_frame_path.split('/')[-1]
      pred_frame_name = os.path.join(pred_sequence_folder, gt_frame_name)
      flow = None
      occlusion = None
      logging.info('Processing sequence %s: frame %s.', sequence, gt_frame_name)
      gt_frame = reader_fn(gt_frame_path, _LABEL_DIVISOR)
      pred_frame = reader_fn(pred_frame_name, _LABEL_DIVISOR)
      if use_optical_flow:
        frame_id = int(os.path.splitext(gt_frame_name)[0])
        flow_path = os.path.join(optical_flow_sequence_folder,
                                 '%06d%s' % (frame_id - 1, _FLOW_EXT))
        occlusion_path = os.path.join(optical_flow_sequence_folder,
                                      '%06d%s' % (frame_id - 1, _OCCLUSION_EXT))
        if tf.io.gfile.exists(flow_path):
          flow = read_numpy_tensor(flow_path)
          occlusion = read_numpy_tensor(occlusion_path)[0, ..., 0]
        else:
          logging.info('Could not find optical flow for current frame.')
          h, w = gt_frame.shape
          flow = np.zeros_like((h, w, 2), np.float32)
          occlusion = np.zeros_like((h, w), np.float32)
      pred_frame = tracker.update(pred_frame, flow, occlusion)
      if FLAGS.output:
        output_folder = os.path.join(FLAGS.output, sequence)
        tf.io.gfile.makedirs(output_folder)
        color_map = vis_utils.save_parsing_result(
            pred_frame,
            _LABEL_DIVISOR,
            thing_classes,
            output_folder,
            os.path.splitext(gt_frame_name)[0],
            color_map,
            colormap_name=colormap_name)
      metric.update_state(
          tf.convert_to_tensor(gt_frame), tf.convert_to_tensor(pred_frame),
          sequence)

  logging.info('Final results:')
  logging.info(_format_output(metric.result()))


if __name__ == '__main__':
  flags.mark_flags_as_required(['gt', 'pred'])
  app.run(main)
