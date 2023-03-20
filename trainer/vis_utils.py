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

"""Utility functions for the visualizer."""
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from deeplab2.data import ade20k_constants
from deeplab2.data import coco_constants
from deeplab2.data import waymo_constants

# Amount of color perturbation added to colormap.
_COLOR_PERTURBATION = 60


def bit_get(val, idx):
  """Gets the bit value.

  Args:
    val: Input value, int or numpy int array.
    idx: Which bit of the input val.

  Returns:
    The "idx"-th bit of input val.
  """
  return (val >> idx) & 1


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((512, 3), dtype=int)
  ind = np.arange(512, dtype=int)

  for shift in reversed(list(range(8))):
    for channel in range(3):
      colormap[:, channel] |= bit_get(ind, channel) << shift
    ind >>= 3

  return colormap


def create_rgb_from_instance_map(instance_map):
  """Creates an RGB image from an instance map for visualization.

  To assign a color to each instance, if the maximum value of the instance
  labels is smaller than the maximum allowed value of Pascal's colormap, we use
  Pascal's colormap. Otherwise, we use random and non-repeated colors.

  Args:
    instance_map: Numpy array of shape `[height, width]`, the instance map.

  Returns:
    instance_image: Numpy array of shape `[height, width, 3]`, the visualized
      RGB instance image.
  """
  # pylint: disable=protected-access
  if np.max(instance_map) < 512:
    colormap = create_pascal_label_colormap()
    instance_image = colormap[instance_map]
  else:
    np.random.seed(0)

    used_colors = [(0, 0, 0)]
    instanc_map_shape = instance_map.shape
    instance_image = np.zeros([instanc_map_shape[0], instanc_map_shape[1], 3],
                              np.uint8)
    instance_ids = np.unique(instance_map)
    for instance_id in instance_ids:
      # We preserve the id "0" for stuff.
      if instance_id == 0:
        continue
      r = np.random.randint(0, 256, dtype=np.uint8)
      g = np.random.randint(0, 256, dtype=np.uint8)
      b = np.random.randint(0, 256, dtype=np.uint8)
      while (r, g, b) in used_colors:
        r = np.random.randint(0, 256, dtype=np.uint8)
        g = np.random.randint(0, 256, dtype=np.uint8)
        b = np.random.randint(0, 256, dtype=np.uint8)
      instance_image[instance_map == instance_id, :] = (r, g, b)
      used_colors.append((r, g, b))
    instance_image[instance_map == 0, :] = (0, 0, 0)

  return instance_image


def _generate_color(used_colors):
  """"Generates a non-repeated color.

  This function first uses the pascal colormap to generate the color. If more
  colors are requested, it randomly generates a non-repeated color.

  Args:
    used_colors: A list, where each element is a tuple in the format of
      (r, g, b).

  Returns:
    A tuple representing a color in the format of (r, g, b).
    A list, which is the updated `used_colors` with the returned color tuple
      appended to it.
  """

  pascal_colormap = create_pascal_label_colormap()

  if len(used_colors) < len(pascal_colormap):
    color = tuple(pascal_colormap[len(used_colors)])
  else:
    r = np.random.randint(0, 256, dtype=np.uint8)
    g = np.random.randint(0, 256, dtype=np.uint8)
    b = np.random.randint(0, 256, dtype=np.uint8)
    while (r, g, b) in used_colors:
      r = np.random.randint(0, 256, dtype=np.uint8)
      g = np.random.randint(0, 256, dtype=np.uint8)
      b = np.random.randint(0, 256, dtype=np.uint8)
    color = (r, g, b)
  used_colors.append(color)

  return color, used_colors


def overlay_heatmap_on_image(heatmap,
                             input_image,
                             dpi=80.0,
                             add_color_bar=False):
  """Overlays a heatmap on top of an image.

  Args:
    heatmap: A numpy array (float32) of shape `[height, width]`,
      which is the heatmap of keypoints.
    input_image: A numpy array (float32 or uint8) of shape
      `[height, width, 3]`, which is an image and all the pixel values are in
      the range of [0.0, 255.0].
    dpi: Float, the dpi of the output image.
    add_color_bar: Boolean, whether to add a colorbar to the output image.

  Returns:
    A numpy array (uint8) of the same shape as the `input_image`.
  """

  # Generate the cmap.
  cmap = plt.cm.Reds  # pytype: disable=module-attr
  # pylint: disable=protected-access
  cmap._init()
  # pylint: disable=protected-access
  cmap._lut[:, -1] = np.linspace(0, 1.0, 259)

  # Plot.
  image = input_image.astype(np.float32) / 255.0
  image_height, image_width, _ = image.shape
  fig, ax = plt.subplots(
      1,
      1,
      facecolor='white',
      figsize=(image_width / dpi, image_height / dpi),
      dpi=dpi)
  grid_y, grid_x = np.mgrid[0:image_height, 0:image_width]
  cb = ax.contourf(grid_x, grid_y, heatmap, 10, cmap=cmap)
  ax.imshow(image)
  ax.grid(False)
  plt.axis('off')
  if add_color_bar:
    plt.colorbar(cb)
  fig.subplots_adjust(bottom=0)
  fig.subplots_adjust(top=1)
  fig.subplots_adjust(right=1)
  fig.subplots_adjust(left=0)

  # Get the output image.
  fig.canvas.draw()
  # pylint: disable=protected-access
  output_image = np.array(fig.canvas.renderer._renderer)[:, :, :-1]
  plt.close()

  return output_image


# pylint: disable=invalid-name
def make_colorwheel():
  """Generates a color wheel for optical flow visualization.

  Reference implementation:
  https://github.com/tomrunia/OpticalFlow_Visualization

  Returns:
    flow_image: A numpy array of output image.
  """

  RY = 15
  YG = 6
  GC = 4
  CB = 11
  BM = 13
  MR = 6

  ncols = RY + YG + GC + CB + BM + MR
  colorwheel = np.zeros((ncols, 3))
  col = 0

  # RY
  colorwheel[0:RY, 0] = 255
  colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
  col = col + RY
  # YG
  colorwheel[col:col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
  colorwheel[col:col + YG, 1] = 255
  col = col + YG
  # GC
  colorwheel[col:col + GC, 1] = 255
  colorwheel[col:col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
  col = col + GC
  # CB
  colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
  colorwheel[col:col+CB, 2] = 255
  col = col+CB
  # BM
  colorwheel[col:col + BM, 2] = 255
  colorwheel[col:col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
  col = col + BM
  # MR
  colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
  colorwheel[col:col+MR, 0] = 255
  return colorwheel
# pylint: enable=invalid-name


def flow_compute_color(u, v):
  """Computes color for 2D flow field.

  Reference implementation:
  https://github.com/tomrunia/OpticalFlow_Visualization

  Args:
    u: A numpy array of horizontal flow.
    v: A numpy array of vertical flow.

  Returns:
    flow_image: A numpy array of output image.
  """

  flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)

  colorwheel = make_colorwheel()  # shape [55x3]
  ncols = colorwheel.shape[0]

  rad = np.sqrt(np.square(u) + np.square(v))
  a = np.arctan2(-v, -u) / np.pi

  fk = (a + 1) / 2 * (ncols - 1)
  k0 = np.floor(fk).astype(np.int32)
  k1 = k0 + 1
  k1[k1 == ncols] = 0
  f = fk - k0

  for i in range(colorwheel.shape[1]):
    tmp = colorwheel[:, i]
    color0 = tmp[k0] / 255.0
    color1 = tmp[k1] / 255.0
    color = (1 - f) * color0 + f * color1

    idx = (rad <= 1)
    color[idx] = 1 - rad[idx] * (1 - color[idx])
    color[~idx] = color[~idx] * 0.75

    # The order is RGB.
    ch_idx = i
    flow_image[:, :, ch_idx] = np.floor(255 * color)

  return flow_image


def flow_to_color(flow_uv, clip_flow=None):
  """Applies color to 2D flow field.

  Reference implementation:
  https://github.com/tomrunia/OpticalFlow_Visualization

  Args:
    flow_uv: A numpy array of flow with shape [Height, Width, 2].
    clip_flow: A float to clip the maximum value for the flow.

  Returns:
    flow_image: A numpy array of output image.

  Raises:
    ValueError: Input flow does not have dimension equals to 3.
    ValueError: Input flow does not have shape [H, W, 2].
  """

  if flow_uv.ndim != 3:
    raise ValueError('Input flow must have three dimensions.')
  if flow_uv.shape[2] != 2:
    raise ValueError('Input flow must have shape [H, W, 2].')

  if clip_flow is not None:
    flow_uv = np.clip(flow_uv, 0, clip_flow)

  u = flow_uv[:, :, 0]
  v = flow_uv[:, :, 1]

  rad = np.sqrt(np.square(u) + np.square(v))
  rad_max = np.max(rad)

  epsilon = 1e-5
  u = u / (rad_max + epsilon)
  v = v / (rad_max + epsilon)

  return flow_compute_color(u, v)


def squeeze_batch_dim_and_convert_to_numpy(input_dict):
  for key in input_dict:
    input_dict[key] = tf.squeeze(input_dict[key], axis=0).numpy()
  return input_dict


def create_cityscapes_label_colormap():
  """Creates a label colormap used in CITYSCAPES segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  return colormap


def create_motchallenge_label_colormap():
  """Creates a label colormap used in MOTChallenge-STEP benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [244, 35, 232]
  colormap[1] = [70, 70, 70]
  colormap[2] = [107, 142, 35]
  colormap[3] = [70, 130, 180]
  colormap[4] = [220, 20, 60]
  colormap[5] = [255, 0, 0]
  colormap[6] = [119, 11, 32]
  return colormap


def create_coco_label_colormap():
  """Creates a label colormap used in COCO dataset.

  Returns:
    A colormap for visualizing segmentation results.
  """
  # Obtain the dictionary mapping original category id to contiguous ones.
  coco_categories = coco_constants.get_coco_reduced_meta()
  colormap = np.zeros((256, 3), dtype=np.uint8)
  for category in coco_categories:
    colormap[category['id']] = category['color']
  return colormap


def create_ade20k_label_colormap() -> np.ndarray:
  """Creates a label colormap used in ADE20K dataset.

  Returns:
    A colormap for visualizing segmentation results.
  """
  ade20k_categories = ade20k_constants.get_ade20k_meta()
  colormap = np.zeros((256, 3), dtype=np.uint8)
  for category in ade20k_categories:
    colormap[category['id']] = category['color']
  return colormap


def create_waymo_label_colormap():
  """Creates a label colormap used in Waymo segmentation dataset.

  Returns:
    A colormap for visualizing segmentation results.
  """
  waymo_categories = waymo_constants.get_waymo_meta()
  colormap = np.zeros((256, 3), dtype=np.uint8)
  for category in waymo_categories:
    colormap[category['id']] = category['color']
  return colormap


def label_to_color_image(label, colormap_name='cityscapes'):
  """Adds color defined by the colormap derived from the dataset to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.
    colormap_name: A string specifying the name of the dataset. Used for
      choosing the right colormap. Currently supported: 'cityscapes',
      'motchallenge'. (Default: 'cityscapes')

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the cityscapes colormap.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label. Got {}'.format(label.shape))

  if np.max(label) >= 256:
    raise ValueError(
        'label value too large: {} >= 256.'.format(np.max(label)))

  if colormap_name == 'cityscapes':
    colormap = create_cityscapes_label_colormap()
  elif colormap_name == 'motchallenge':
    colormap = create_motchallenge_label_colormap()
  elif colormap_name == 'coco':
    colormap = create_coco_label_colormap()
  elif colormap_name == 'ade20k':
    colormap = create_ade20k_label_colormap()
  elif colormap_name == 'waymo':
    colormap = create_waymo_label_colormap()
  else:
    raise ValueError('Could not find a colormap for dataset %s.' %
                     colormap_name)
  return colormap[label]


def save_parsing_result(parsing_result,
                        label_divisor,
                        thing_list,
                        save_dir,
                        filename,
                        id_to_colormap=None,
                        colormap_name='cityscapes'):
  """Saves the parsing results.

  The parsing result encodes both semantic segmentation and instance
  segmentation results. In order to visualize the parsing result with only
  one png file, we adopt the following procedures, similar to the
  `visualization.py` provided in the COCO panoptic segmentation evaluation
  codes.

  1. Pixels predicted as `stuff` will take the same semantic color defined
    in the colormap.
  2. Pixels of a predicted `thing` instance will take similar semantic color
    defined in the colormap. For example, `car` class takes blue color in
    the colormap. Predicted car instance 1 will then be colored with the
    blue color perturbed with a small amount of RGB noise.

  Args:
    parsing_result: The numpy array to be saved. The data will be converted
      to uint8 and saved as png image.
    label_divisor: Integer, encoding the semantic segmentation and instance
      segmentation results as value = semantic_label * label_divisor +
      instance_label.
    thing_list: A list containing the semantic indices of the thing classes.
    save_dir: String, the directory to which the results will be saved.
    filename: String, the image filename.
    id_to_colormap: An optional mapping from track ID to color.
    colormap_name: A string specifying the dataset to choose the corresponding
      color map. Currently supported: 'cityscapes', 'motchallenge'. (Default:
      'cityscapes').

  Raises:
    ValueError: If parsing_result is not of rank 2 or its value in semantic
      segmentation result is larger than color map maximum entry.
    ValueError: If provided colormap_name is not supported.

  Returns:
    If id_to_colormap is passed, the updated id_to_colormap will be returned.
  """
  if parsing_result.ndim != 2:
    raise ValueError('Expect 2-D parsing result. Got {}'.format(
        parsing_result.shape))
  semantic_result = parsing_result // label_divisor
  instance_result = parsing_result % label_divisor
  colormap_max_value = 256
  if np.max(semantic_result) >= colormap_max_value:
    raise ValueError('Predicted semantic value too large: {} >= {}.'.format(
        np.max(semantic_result), colormap_max_value))
  height, width = parsing_result.shape
  colored_output = np.zeros((height, width, 3), dtype=np.uint8)
  if colormap_name == 'cityscapes':
    colormap = create_cityscapes_label_colormap()
  elif colormap_name == 'motchallenge':
    colormap = create_motchallenge_label_colormap()
  elif colormap_name == 'coco':
    colormap = create_coco_label_colormap()
  elif colormap_name == 'ade20k':
    colormap = create_ade20k_label_colormap()
  elif colormap_name == 'waymo':
    colormap = create_waymo_label_colormap()
  else:
    raise ValueError('Could not find a colormap for dataset %s.' %
                     colormap_name)
  # Keep track of used colors.
  used_colors = set()
  if id_to_colormap is not None:
    used_colors = set([tuple(val) for val in id_to_colormap.values()])
    np_state = None
  else:
    # Use random seed 0 in order to reproduce the same visualization.
    np_state = np.random.RandomState(0)

  unique_semantic_values = np.unique(semantic_result)
  for semantic_value in unique_semantic_values:
    semantic_mask = semantic_result == semantic_value
    if semantic_value in thing_list:
      # For `thing` class, we will add a small amount of random noise to its
      # correspondingly predefined semantic segmentation colormap.
      unique_instance_values = np.unique(instance_result[semantic_mask])
      for instance_value in unique_instance_values:
        instance_mask = np.logical_and(semantic_mask,
                                       instance_result == instance_value)
        if id_to_colormap is not None:
          if instance_value in id_to_colormap:
            colored_output[instance_mask] = id_to_colormap[instance_value]
            continue
        random_color = perturb_color(
            colormap[semantic_value],
            _COLOR_PERTURBATION,
            used_colors,
            random_state=np_state)
        colored_output[instance_mask] = random_color
        if id_to_colormap is not None:
          id_to_colormap[instance_value] = random_color
    else:
      # For `stuff` class, we use the defined semantic color.
      colored_output[semantic_mask] = colormap[semantic_value]
      used_colors.add(tuple(colormap[semantic_value]))

  pil_image = PIL.Image.fromarray(colored_output.astype(dtype=np.uint8))
  with tf.io.gfile.GFile('{}/{}.png'.format(save_dir, filename), mode='w') as f:
    pil_image.save(f, 'PNG')
  if id_to_colormap is not None:
    return id_to_colormap


def perturb_color(color,
                  noise,
                  used_colors=None,
                  max_trials=50,
                  random_state=None):
  """Pertrubs the color with some noise.

  If `used_colors` is not None, we will return the color that has
  not appeared before in it.

  Args:
    color: A numpy array with three elements [R, G, B].
    noise: Integer, specifying the amount of perturbing noise.
    used_colors: A set, used to keep track of used colors.
    max_trials: An integer, maximum trials to generate random color.
    random_state: An optional np.random.RandomState. If passed, will be used to
      generate random numbers.

  Returns:
    A perturbed color that has not appeared in used_colors.
  """
  for _ in range(max_trials):
    if random_state is not None:
      random_color = color + random_state.randint(
          low=-noise, high=noise + 1, size=3)
    else:
      random_color = color + np.random.randint(low=-noise,
                                               high=noise+1,
                                               size=3)
    random_color = np.maximum(0, np.minimum(255, random_color))
    if used_colors is None:
      return random_color
    elif tuple(random_color) not in used_colors:
      used_colors.add(tuple(random_color))
      return random_color
  logging.warning('Using duplicate random color.')
  return random_color


def save_annotation(label,
                    save_dir,
                    filename,
                    add_colormap=True,
                    normalize_to_unit_values=False,
                    scale_factor=None,
                    colormap_name='cityscapes',
                    output_dtype=np.uint8):
  """Saves the given label to image on disk.

  Args:
    label: The numpy array to be saved. The data will be converted
      to uint8 and saved as png image.
    save_dir: String, the directory to which the results will be saved.
    filename: String, the image filename.
    add_colormap: Boolean, add color map to the label or not.
    normalize_to_unit_values: Boolean, normalize the input values to [0, 1].
    scale_factor: Float or None, the factor to scale the input values.
    colormap_name: A string specifying the dataset to choose the corresponding
      color map. Currently supported: 'cityscapes', 'motchallenge'. (Default:
      'cityscapes').
    output_dtype: The numpy dtype of output before converting to PIL image.
  """
  # Add colormap for visualizing the prediction.
  if add_colormap:
    colored_label = label_to_color_image(label, colormap_name)
  else:
    colored_label = label
    if normalize_to_unit_values:
      min_value = np.amin(colored_label)
      max_value = np.amax(colored_label)
      range_value = max_value - min_value
      if range_value != 0:
        colored_label = (colored_label - min_value) / range_value

    if scale_factor:
      colored_label = scale_factor * colored_label

  pil_image = PIL.Image.fromarray(colored_label.astype(dtype=output_dtype))
  with tf.io.gfile.GFile('%s/%s.png' % (save_dir, filename), mode='w') as f:
    pil_image.save(f, 'PNG')
