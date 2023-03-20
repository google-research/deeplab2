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

# pylint: disable=line-too-long
# pyformat: disable
r"""Creates a JSON file with info for a split of Cityscapes images.

This single-purpose version has special handling for the directory structure of
CityScapes dataset and the expected output ids.

Sample commands:

python create_images_json_for_cityscapes.py \
  --image_dir=${DATA_ROOT}/leftImg8bit/${IMAGES_SPLIT} \
  --output_json_path=${PATH_TO_SAVE}/${IMAGES_SPLIT}_images.json \
  --only_basename \
  --include_image_type_suffix=false
"""
# pyformat: enable
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import re

from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'image_dir', None,
    'The top-level directory of image files to be included in the set.')

flags.DEFINE_list(
    'keep_cities', None,
    'Comma-separated list of strings specifying cities to be processed.')

flags.DEFINE_string('output_json_path', None,
                    'Output path to which is written the image info JSON.')

flags.DEFINE_boolean(
    'only_basename', True,
    'If set, the included "file_name" properties of the images in the JSON '
    'file will only include the base name and not the city directory. Used for '
    'tools that do not support nested directories.')

flags.DEFINE_boolean(
    'include_image_type_suffix', True,
    'If set, will include the suffix of the image type (e.g. "_leftImg8bit") '
    'in the "file_name" properties of the image.')


def _create_images_json(image_dir, output_json_path, only_basename=False,
                        include_image_type_suffix=True, keep_cities=None):
  """Lists the images in image_dir and writes out the info JSON for them."""
  images_info_array = []
  for city_dir in tf.io.gfile.listdir(image_dir):
    if keep_cities and city_dir not in keep_cities:
      continue
    image_id_re = r'%s_[0-9]+_[0-9]+' % city_dir
    image_id_re = re.compile(image_id_re)
    for image_basename in tf.io.gfile.listdir(
        os.path.join(image_dir, city_dir)):
      match = image_id_re.match(image_basename)
      image_id = image_basename[match.start():match.end()]
      if include_image_type_suffix:
        file_name = image_basename
      else:
        file_name = image_id + os.path.splitext(image_basename)[1]
      if not only_basename:
        file_name = os.path.join(city_dir, file_name)
      image_info_dict = {'id': image_id, 'file_name': file_name}
      images_info_array.append(image_info_dict)

  info_dict = {'images': images_info_array}

  with tf.io.gfile.GFile(output_json_path, 'w+') as json_file:
    json.dump(info_dict, json_file)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  keep_cities = None
  if FLAGS.keep_cities:
    keep_cities = [str(x) for x in FLAGS.keep_cities]
  _create_images_json(
      FLAGS.image_dir,
      FLAGS.output_json_path,
      only_basename=FLAGS.only_basename,
      include_image_type_suffix=FLAGS.include_image_type_suffix,
      keep_cities=keep_cities)


if __name__ == '__main__':
  flags.mark_flags_as_required(['image_dir', 'output_json_path'])
  app.run(main)
