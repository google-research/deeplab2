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

"""Meta info of Waymo Open Dataset: Panoramic Video Panoptic Segmentation.

Dataset website: https://waymo.com/open/
GitHub: https://github.com/waymo-research/waymo-open-dataset

References:

- Jieru Mei, Alex Zihao Zhu, Xinchen Yan, Hang Yan, Siyuan Qiao, Yukun Zhu,
Liang-Chieh Chen, Henrik Kretzschmar, Dragomir Anguelov. "Waymo Open Dataset:
Panoramic Video Panoptic Segmentation." In ECCV, 2022.
"""

from typing import Any, Sequence

import immutabledict

COLORMAP = "waymo"
PANOPTIC_LABEL_DIVISOR = 100000

IGNORE_LABEL_NAME = "unknown"
IGNORE_LABEL = 0

_WAYMO_COLORS = immutabledict.immutabledict({
    "bicycle": [119, 11, 32],
    "bird": [127, 96, 0],
    "building": [70, 70, 70],
    "bus": [0, 60, 100],
    "car": [0, 0, 142],
    "construction_cone_pole": [230, 145, 56],
    "cyclist": [255, 0, 0],
    "dynamic": [102, 102, 102],
    "ground": [102, 102, 102],
    "ground_animal": [91, 15, 0],
    "lane_marker": [234, 209, 220],
    "motorcycle": [0, 0, 230],
    "motorcyclist": [180, 0, 0],
    "other_large_vehicle": [61, 133, 198],
    "other_pedestrian_object": [234, 153, 153],
    "person": [220, 20, 60],
    "pole": [153, 153, 153],
    "road": [128, 64, 128],
    "road_marker": [217, 210, 233],
    "sdc": [102, 102, 102],
    "sidewalk": [244, 35, 232],
    "sign": [246, 178, 107],
    "sky": [70, 130, 180],
    "static": [102, 102, 102],
    "traffic_light": [250, 170, 30],
    "trailer": [111, 168, 220],
    "truck": [0, 0, 70],
    "unknown": [102, 102, 102],
    "vegetation": [107, 142, 35],
})

_WAYMO_CLASS_NAMES = [
    "unknown",
    "sdc",
    "car",
    "truck",
    "bus",
    "other_large_vehicle",
    "bicycle",
    "motorcycle",
    "trailer",
    "person",
    "cyclist",
    "motorcyclist",
    "bird",
    "ground_animal",
    "construction_cone_pole",
    "pole",
    "other_pedestrian_object",
    "sign",
    "traffic_light",
    "building",
    "road",
    "lane_marker",
    "road_marker",
    "sidewalk",
    "vegetation",
    "sky",
    "ground",
    "dynamic",
    "static",
]

_IS_THINGS = [
    "car", "truck", "bus", "other_large_vehicle", "trailer", "person",
    "cyclist", "motorcyclist"
]


def get_waymo_meta() -> Sequence[Any]:
  """Gets the meta info for waymo dataset."""
  meta = []
  for name_id, name in enumerate(_WAYMO_CLASS_NAMES):
    item = {
        "color": _WAYMO_COLORS[name],
        "name": name,
        "id": name_id,
        "isthing": int(name in _IS_THINGS)
    }
    meta.append(item)
  return meta
