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

r"""Creates STEP panoptic map from semantic and instance maps.

This script implements the process of merging semantic maps (from our extra
annotations[1]) and instance maps (collected from the MOTS[2]) to obtain the
STEP panoptic map.

[1] Mark Weber, etc. STEP: Segmenting and Tracking Every Pixel, arXiv:2102.11859
[2] Paul Voigtlaender, etc. Multi-object tracking and segmentation. CVPR, 2019

To run this script, you need to install opencv-python (>=4.4.0).
e.g. In Linux, run
$pip install opencv-python

The input directory structure should be as follows:

+ INPUT_SEMANTIC_MAP_ROOT_DIR
  + train
    + sequence_id
      - *.png
      ...
  + val

+ INPUT_INSTANCE_MAP_ROOT_DIR
  + train
    + sequence_id
      - *.png
       ...
  + val

+ OUTPUT_PANOPTIC_MAP_ROOT_DIR (generated)
  + train
    + sequence_id
      - *.png
       ...
  + val

The ground-truth panoptic map is generated and encoded as the following in PNG
format:
  R: semantic_id
  G: instance_id // 256
  B: instance % 256

The generated panoptic maps will be used by ../build_step_data.py to create
tfrecords for training and evaluation.

Example to run the scipt:

```bash
   python deeplab2/data/utils/create_step_panoptic_maps.py \
     --input_semantic_map_root_dir=...
     ...
```
"""

import os
from typing import Any, Sequence, Union

from absl import app
from absl import flags
from absl import logging
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

FLAGS = flags.FLAGS
flags.DEFINE_string('input_semantic_map_root_dir', None,
                    'Path to a directory containing the semantic map.')
flags.DEFINE_string('input_instance_root_dir', None,
                    'Path to a directory containing the instance map.')
flags.DEFINE_string('output_panoptic_map_root_dir', None,
                    'Path to a directory where we write the panoptic map.')
flags.DEFINE_integer(
    'kernel_size', 15, 'Kernel size to extend instance object boundary when '
    'merging it with semantic map.')
flags.DEFINE_enum('dataset_name', 'kitti-step',
                  ['kitti-step', 'motchallenge-step'], 'Name of the dataset')

# The label definition below follows Cityscapes label definition in
# https://www.cityscapes-dataset.com/.
MOTCHALLENGE_MERGED_CLASSES = (0, 3, 4, 5, 6, 7, 9, 13, 14, 15, 16, 17)
NUM_VALID_CLASSES = 19
SEMANTIC_CAR = 13
SEMANTIC_PERSON = 11
SEMANTIC_VOID = 255
INSTANCE_CAR = 1
INSTANCE_PERSON = 2
INSTANCE_LABEL_DIVISOR = 1000


def encode_panoptic_map(panoptic_map: np.ndarray) -> np.ndarray:
  """Encodes the panoptic map in three channel image format."""
  # Encoding format: R: semantic | G: instance // 256 | B: instance % 256
  semantic_id = panoptic_map // INSTANCE_LABEL_DIVISOR
  instance_id = panoptic_map % INSTANCE_LABEL_DIVISOR
  return np.dstack(
      (semantic_id, instance_id // 256, instance_id % 256)).astype(np.uint8)


def load_image(image_path: str) -> np.ndarray:
  """Loads an image as numpy array."""
  with tf.io.gfile.GFile(image_path, 'rb') as f:
    return np.array(Image.open(f))


def _update_motchallege_label_map(semantic_map: np.ndarray) -> np.ndarray:
  """Updates semantic map by merging some classes."""
  # For MOTChallenge dataset, we merge some classes since they are less
  # representative:
  #--------------------------------------------------------------
  # Original index | Updated index|          Note
  #----------------+--------------+------------------------------
  #       0        |      1       |   map road to sidewalk
  #       1        |      1       |   keep sidewalk
  #       2        |      2       |   keep building
  #       3        |     255      |   not present anyway
  #       4        |     255      |   remove fence
  #       5        |     255      |   remove pole
  #       6        |     255      |   remove traffic light
  #       7        |     255      |   not present anyway
  #       8        |      8       |   keep vegetation
  #       9        |      8       |   map terrain to vegetation
  #       10       |     10       |   keep sky
  #       11       |     11       |   keep pedestrain
  #       12       |     12       |   keep rider
  #       13       |     255      |   remove car
  #       14       |     255      |   not present anyway
  #       15       |     255      |   not present anyway
  #       16       |     255      |   not present anyway
  #       17       |     255      |   remove motorcycle
  #       18       |     18       |   keep bicycle
  #       255      |     255      |   keep void
  #--------------------------------------------------------------
  for label in MOTCHALLENGE_MERGED_CLASSES:
    if label == 0:
      semantic_map[semantic_map == label] = 1
    elif label == 9:
      semantic_map[semantic_map == label] = 8
    else:
      semantic_map[semantic_map == label] = 255
  return semantic_map


def _compute_panoptic_id(semantic_id: Union[int, np.ndarray],
                         instance_id: Union[int, np.ndarray]) -> Any:
  """Gets the panoptic id by combining semantic and instance id."""
  return semantic_id * INSTANCE_LABEL_DIVISOR + instance_id


def _remap_motchallege_semantic_indices(panoptic_id: np.ndarray) -> np.ndarray:
  """Updates MOTChallenge semantic map by re-mapping label indices."""
  semantic_id = panoptic_id // INSTANCE_LABEL_DIVISOR
  instance_id = panoptic_id % INSTANCE_LABEL_DIVISOR
  # Re-mapping index
  # 1 -> 0:     sidewalk
  # 2 -> 1:     building
  # 8 -> 2:     vegetation
  # 10 -> 3:    sky
  # 11 -> 4:    pedestrain
  # 12 -> 5:    rider
  # 18 -> 6:    bicycle
  # 255 -> 255: void
  all_labels = set(range(NUM_VALID_CLASSES))
  for i, label in enumerate(
      sorted(all_labels - set(MOTCHALLENGE_MERGED_CLASSES))):
    semantic_id[semantic_id == label] = i
  return _compute_panoptic_id(semantic_id, instance_id)


def _get_semantic_maps(semantic_map_root: str, dataset_split: str,
                       sequence_id: str) -> Sequence[str]:
  """Gets files for the specified data type and dataset split."""
  search_files = os.path.join(semantic_map_root, dataset_split, sequence_id,
                              '*')
  filenames = tf.io.gfile.glob(search_files)
  return sorted(filenames)


class StepPanopticMapGenerator(object):
  """Class to generate and write panoptic map from semantic and instance map."""

  def __init__(self, kernel_size: int, dataset_name: str):
    self.kernel_size = kernel_size
    self.is_mots_challenge = (dataset_name == 'motchallenge-step')

  def _update_semantic_label_map(self, instance_map: np.ndarray,
                                 semantic_map: np.ndarray) -> np.ndarray:
    """Updates semantic map by leveraging semantic map and instance map."""
    kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)
    updated_semantic_map = semantic_map.astype(np.int32)
    if self.is_mots_challenge:
      updated_semantic_map = _update_motchallege_label_map(updated_semantic_map)
    for label in (SEMANTIC_CAR, SEMANTIC_PERSON):
      semantic_mask = (semantic_map == label)
      if label == SEMANTIC_PERSON:
        # The instance ids are encoded according to
        # https://www.vision.rwth-aachen.de/page/mots
        instance_mask = (
            instance_map // INSTANCE_LABEL_DIVISOR == INSTANCE_PERSON)
      elif label == SEMANTIC_CAR:
        instance_mask = instance_map // INSTANCE_LABEL_DIVISOR == INSTANCE_CAR
      # Run dilation on the instance map to merge it with semantic map.
      instance_mask = instance_mask.astype(np.uint8)
      dilated_instance_mask = cv2.dilate(instance_mask, kernel)
      void_boundary = np.logical_and(dilated_instance_mask - instance_mask,
                                     semantic_mask)
      updated_semantic_map[void_boundary] = SEMANTIC_VOID
    return updated_semantic_map

  def merge_panoptic_map(self, semantic_map: np.ndarray,
                         instance_map: np.ndarray) -> np.ndarray:
    """Merges semantic labels with given instance map."""
    # Use semantic_map as the base map.
    updated_semantic_map = self._update_semantic_label_map(
        instance_map, semantic_map)
    panoptic_map = _compute_panoptic_id(updated_semantic_map, 0)
    # Merge instance.
    mask_car = instance_map // INSTANCE_LABEL_DIVISOR == INSTANCE_CAR
    # The instance map has index from 0 but the panoptic map's instance index
    # will start from 1.
    instance_id = (instance_map[mask_car] % INSTANCE_LABEL_DIVISOR) + 1
    panoptic_map[mask_car] = _compute_panoptic_id(SEMANTIC_CAR,
                                                  instance_id.astype(np.int32))
    mask_person = instance_map // INSTANCE_LABEL_DIVISOR == INSTANCE_PERSON
    instance_id = (instance_map[mask_person] % INSTANCE_LABEL_DIVISOR) + 1
    panoptic_map[mask_person] = _compute_panoptic_id(
        SEMANTIC_PERSON, instance_id.astype(np.int32))

    # Remap label indices.
    if self.is_mots_challenge:
      panoptic_map = _remap_motchallege_semantic_indices(panoptic_map)
    return panoptic_map

  def build_panoptic_maps(self, semantic_map_root: str, instance_map_root: str,
                          dataset_split: str, sequence_id: str,
                          panoptic_map_root: str):
    """Creates panoptic maps and save them as PNG format.

    Args:
      semantic_map_root: Semantic map root folder.
      instance_map_root: Instance map root folder.
      dataset_split: Train/Val/Test split of the data.
      sequence_id: Sequence id of the data.
      panoptic_map_root: Panoptic map root folder where the encoded panoptic
        maps will be saved.
    """
    semantic_maps = _get_semantic_maps(semantic_map_root, dataset_split,
                                       sequence_id)
    for semantic_map_path in semantic_maps:
      image_name = os.path.basename(semantic_map_path)
      instance_map_path = os.path.join(instance_map_root, dataset_split,
                                       sequence_id, image_name)
      if not tf.io.gfile.exists(instance_map_path):
        logging.warn('Could not find instance map for %s', semantic_map_path)
        continue
      semantic_map = load_image(semantic_map_path)
      instance_map = load_image(instance_map_path)
      panoptic_map = self.merge_panoptic_map(semantic_map, instance_map)
      encoded_panoptic_map = Image.fromarray(
          encode_panoptic_map(panoptic_map)).convert('RGB')
      panoptic_map_path = os.path.join(panoptic_map_root, dataset_split,
                                       sequence_id, image_name)
      with tf.io.gfile.GFile(panoptic_map_path, 'wb') as f:
        encoded_panoptic_map.save(f, format='PNG')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  panoptic_map_generator = StepPanopticMapGenerator(FLAGS.kernel_size,
                                                    FLAGS.dataset_name)
  for dataset_split in ('train', 'val', 'test'):
    sem_dir = os.path.join(FLAGS.input_semantic_map_root_dir, dataset_split)
    if not tf.io.gfile.exists(sem_dir):
      logging.info('Split %s not found.', dataset_split)
      continue
    for set_dir in tf.io.gfile.listdir(sem_dir):
      tf.io.gfile.makedirs(
          os.path.join(FLAGS.output_panoptic_map_root_dir, dataset_split,
                       set_dir))
      logging.info('Start to create panoptic map for split %s, sequence %s.',
                   dataset_split, set_dir)
      panoptic_map_generator.build_panoptic_maps(
          FLAGS.input_semantic_map_root_dir, FLAGS.input_instance_root_dir,
          dataset_split, set_dir, FLAGS.output_panoptic_map_root_dir)


if __name__ == '__main__':
  app.run(main)
