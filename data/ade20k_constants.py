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

"""File containing the meta info of ADE20k dataset."""

import copy
from typing import Sequence, Any

_ADE20K_META = [{
    "color": [0, 0, 0],
    "isthing": 0,
    "name": "other-objects",
    "id": 0
}, {
    "color": [120, 120, 120],
    "isthing": 0,
    "name": "wall",
    "id": 1
}, {
    "color": [180, 120, 120],
    "isthing": 0,
    "name": "building",
    "id": 2
}, {
    "color": [6, 230, 230],
    "isthing": 0,
    "name": "sky",
    "id": 3
}, {
    "color": [80, 50, 50],
    "isthing": 0,
    "name": "floor",
    "id": 4
}, {
    "color": [4, 200, 3],
    "isthing": 0,
    "name": "tree",
    "id": 5
}, {
    "color": [120, 120, 80],
    "isthing": 0,
    "name": "ceiling",
    "id": 6
}, {
    "color": [140, 140, 140],
    "isthing": 0,
    "name": "road",
    "id": 7
}, {
    "color": [204, 5, 255],
    "isthing": 1,
    "name": "bed",
    "id": 8
}, {
    "color": [230, 230, 230],
    "isthing": 1,
    "name": "windowpane",
    "id": 9
}, {
    "color": [4, 250, 7],
    "isthing": 0,
    "name": "grass",
    "id": 10
}, {
    "color": [224, 5, 255],
    "isthing": 1,
    "name": "cabinet",
    "id": 11
}, {
    "color": [235, 255, 7],
    "isthing": 0,
    "name": "sidewalk",
    "id": 12
}, {
    "color": [150, 5, 61],
    "isthing": 1,
    "name": "person",
    "id": 13
}, {
    "color": [120, 120, 70],
    "isthing": 0,
    "name": "earth",
    "id": 14
}, {
    "color": [8, 255, 51],
    "isthing": 1,
    "name": "door",
    "id": 15
}, {
    "color": [255, 6, 82],
    "isthing": 1,
    "name": "table",
    "id": 16
}, {
    "color": [143, 255, 140],
    "isthing": 0,
    "name": "mountain",
    "id": 17
}, {
    "color": [204, 255, 4],
    "isthing": 0,
    "name": "plant",
    "id": 18
}, {
    "color": [255, 51, 7],
    "isthing": 1,
    "name": "curtain",
    "id": 19
}, {
    "color": [204, 70, 3],
    "isthing": 1,
    "name": "chair",
    "id": 20
}, {
    "color": [0, 102, 200],
    "isthing": 1,
    "name": "car",
    "id": 21
}, {
    "color": [61, 230, 250],
    "isthing": 0,
    "name": "water",
    "id": 22
}, {
    "color": [255, 6, 51],
    "isthing": 1,
    "name": "painting",
    "id": 23
}, {
    "color": [11, 102, 255],
    "isthing": 1,
    "name": "sofa",
    "id": 24
}, {
    "color": [255, 7, 71],
    "isthing": 1,
    "name": "shelf",
    "id": 25
}, {
    "color": [255, 9, 224],
    "isthing": 0,
    "name": "house",
    "id": 26
}, {
    "color": [9, 7, 230],
    "isthing": 0,
    "name": "sea",
    "id": 27
}, {
    "color": [220, 220, 220],
    "isthing": 1,
    "name": "mirror",
    "id": 28
}, {
    "color": [255, 9, 92],
    "isthing": 0,
    "name": "rug",
    "id": 29
}, {
    "color": [112, 9, 255],
    "isthing": 0,
    "name": "field",
    "id": 30
}, {
    "color": [8, 255, 214],
    "isthing": 1,
    "name": "armchair",
    "id": 31
}, {
    "color": [7, 255, 224],
    "isthing": 1,
    "name": "seat",
    "id": 32
}, {
    "color": [255, 184, 6],
    "isthing": 1,
    "name": "fence",
    "id": 33
}, {
    "color": [10, 255, 71],
    "isthing": 1,
    "name": "desk",
    "id": 34
}, {
    "color": [255, 41, 10],
    "isthing": 0,
    "name": "rock",
    "id": 35
}, {
    "color": [7, 255, 255],
    "isthing": 1,
    "name": "wardrobe",
    "id": 36
}, {
    "color": [224, 255, 8],
    "isthing": 1,
    "name": "lamp",
    "id": 37
}, {
    "color": [102, 8, 255],
    "isthing": 1,
    "name": "bathtub",
    "id": 38
}, {
    "color": [255, 61, 6],
    "isthing": 1,
    "name": "railing",
    "id": 39
}, {
    "color": [255, 194, 7],
    "isthing": 1,
    "name": "cushion",
    "id": 40
}, {
    "color": [255, 122, 8],
    "isthing": 0,
    "name": "base",
    "id": 41
}, {
    "color": [0, 255, 20],
    "isthing": 1,
    "name": "box",
    "id": 42
}, {
    "color": [255, 8, 41],
    "isthing": 1,
    "name": "column",
    "id": 43
}, {
    "color": [255, 5, 153],
    "isthing": 1,
    "name": "signboard",
    "id": 44
}, {
    "color": [6, 51, 255],
    "isthing": 1,
    "name": "chest-of-drawers",
    "id": 45
}, {
    "color": [235, 12, 255],
    "isthing": 1,
    "name": "counter",
    "id": 46
}, {
    "color": [160, 150, 20],
    "isthing": 0,
    "name": "sand",
    "id": 47
}, {
    "color": [0, 163, 255],
    "isthing": 1,
    "name": "sink",
    "id": 48
}, {
    "color": [140, 140, 140],
    "isthing": 0,
    "name": "skyscraper",
    "id": 49
}, {
    "color": [250, 10, 15],
    "isthing": 1,
    "name": "fireplace",
    "id": 50
}, {
    "color": [20, 255, 0],
    "isthing": 1,
    "name": "refrigerator",
    "id": 51
}, {
    "color": [31, 255, 0],
    "isthing": 0,
    "name": "grandstand",
    "id": 52
}, {
    "color": [255, 31, 0],
    "isthing": 0,
    "name": "path",
    "id": 53
}, {
    "color": [255, 224, 0],
    "isthing": 1,
    "name": "stairs",
    "id": 54
}, {
    "color": [153, 255, 0],
    "isthing": 0,
    "name": "runway",
    "id": 55
}, {
    "color": [0, 0, 255],
    "isthing": 1,
    "name": "case",
    "id": 56
}, {
    "color": [255, 71, 0],
    "isthing": 1,
    "name": "pool-table",
    "id": 57
}, {
    "color": [0, 235, 255],
    "isthing": 1,
    "name": "pillow",
    "id": 58
}, {
    "color": [0, 173, 255],
    "isthing": 1,
    "name": "screen-door",
    "id": 59
}, {
    "color": [31, 0, 255],
    "isthing": 0,
    "name": "stairway",
    "id": 60
}, {
    "color": [11, 200, 200],
    "isthing": 0,
    "name": "river",
    "id": 61
}, {
    "color": [255, 82, 0],
    "isthing": 0,
    "name": "bridge",
    "id": 62
}, {
    "color": [0, 255, 245],
    "isthing": 1,
    "name": "bookcase",
    "id": 63
}, {
    "color": [0, 61, 255],
    "isthing": 0,
    "name": "blind",
    "id": 64
}, {
    "color": [0, 255, 112],
    "isthing": 1,
    "name": "coffee-table",
    "id": 65
}, {
    "color": [0, 255, 133],
    "isthing": 1,
    "name": "toilet",
    "id": 66
}, {
    "color": [255, 0, 0],
    "isthing": 1,
    "name": "flower",
    "id": 67
}, {
    "color": [255, 163, 0],
    "isthing": 1,
    "name": "book",
    "id": 68
}, {
    "color": [255, 102, 0],
    "isthing": 0,
    "name": "hill",
    "id": 69
}, {
    "color": [194, 255, 0],
    "isthing": 1,
    "name": "bench",
    "id": 70
}, {
    "color": [0, 143, 255],
    "isthing": 1,
    "name": "countertop",
    "id": 71
}, {
    "color": [51, 255, 0],
    "isthing": 1,
    "name": "stove",
    "id": 72
}, {
    "color": [0, 82, 255],
    "isthing": 1,
    "name": "palm",
    "id": 73
}, {
    "color": [0, 255, 41],
    "isthing": 1,
    "name": "kitchen-island",
    "id": 74
}, {
    "color": [0, 255, 173],
    "isthing": 1,
    "name": "computer",
    "id": 75
}, {
    "color": [10, 0, 255],
    "isthing": 1,
    "name": "swivel-chair",
    "id": 76
}, {
    "color": [173, 255, 0],
    "isthing": 1,
    "name": "boat",
    "id": 77
}, {
    "color": [0, 255, 153],
    "isthing": 0,
    "name": "bar",
    "id": 78
}, {
    "color": [255, 92, 0],
    "isthing": 1,
    "name": "arcade-machine",
    "id": 79
}, {
    "color": [255, 0, 255],
    "isthing": 0,
    "name": "hovel",
    "id": 80
}, {
    "color": [255, 0, 245],
    "isthing": 1,
    "name": "bus",
    "id": 81
}, {
    "color": [255, 0, 102],
    "isthing": 1,
    "name": "towel",
    "id": 82
}, {
    "color": [255, 173, 0],
    "isthing": 1,
    "name": "light",
    "id": 83
}, {
    "color": [255, 0, 20],
    "isthing": 1,
    "name": "truck",
    "id": 84
}, {
    "color": [255, 184, 184],
    "isthing": 0,
    "name": "tower",
    "id": 85
}, {
    "color": [0, 31, 255],
    "isthing": 1,
    "name": "chandelier",
    "id": 86
}, {
    "color": [0, 255, 61],
    "isthing": 1,
    "name": "awning",
    "id": 87
}, {
    "color": [0, 71, 255],
    "isthing": 1,
    "name": "streetlight",
    "id": 88
}, {
    "color": [255, 0, 204],
    "isthing": 1,
    "name": "booth",
    "id": 89
}, {
    "color": [0, 255, 194],
    "isthing": 1,
    "name": "television",
    "id": 90
}, {
    "color": [0, 255, 82],
    "isthing": 1,
    "name": "airplane",
    "id": 91
}, {
    "color": [0, 10, 255],
    "isthing": 0,
    "name": "dirt-track",
    "id": 92
}, {
    "color": [0, 112, 255],
    "isthing": 1,
    "name": "apparel",
    "id": 93
}, {
    "color": [51, 0, 255],
    "isthing": 1,
    "name": "pole",
    "id": 94
}, {
    "color": [0, 194, 255],
    "isthing": 0,
    "name": "land",
    "id": 95
}, {
    "color": [0, 122, 255],
    "isthing": 1,
    "name": "bannister",
    "id": 96
}, {
    "color": [0, 255, 163],
    "isthing": 0,
    "name": "escalator",
    "id": 97
}, {
    "color": [255, 153, 0],
    "isthing": 1,
    "name": "ottoman",
    "id": 98
}, {
    "color": [0, 255, 10],
    "isthing": 1,
    "name": "bottle",
    "id": 99
}, {
    "color": [255, 112, 0],
    "isthing": 0,
    "name": "buffet",
    "id": 100
}, {
    "color": [143, 255, 0],
    "isthing": 0,
    "name": "poster",
    "id": 101
}, {
    "color": [82, 0, 255],
    "isthing": 0,
    "name": "stage",
    "id": 102
}, {
    "color": [163, 255, 0],
    "isthing": 1,
    "name": "van",
    "id": 103
}, {
    "color": [255, 235, 0],
    "isthing": 1,
    "name": "ship",
    "id": 104
}, {
    "color": [8, 184, 170],
    "isthing": 1,
    "name": "fountain",
    "id": 105
}, {
    "color": [133, 0, 255],
    "isthing": 0,
    "name": "conveyer-belt",
    "id": 106
}, {
    "color": [0, 255, 92],
    "isthing": 0,
    "name": "canopy",
    "id": 107
}, {
    "color": [184, 0, 255],
    "isthing": 1,
    "name": "washer",
    "id": 108
}, {
    "color": [255, 0, 31],
    "isthing": 1,
    "name": "plaything",
    "id": 109
}, {
    "color": [0, 184, 255],
    "isthing": 0,
    "name": "swimming-pool",
    "id": 110
}, {
    "color": [0, 214, 255],
    "isthing": 1,
    "name": "stool",
    "id": 111
}, {
    "color": [255, 0, 112],
    "isthing": 1,
    "name": "barrel",
    "id": 112
}, {
    "color": [92, 255, 0],
    "isthing": 1,
    "name": "basket",
    "id": 113
}, {
    "color": [0, 224, 255],
    "isthing": 0,
    "name": "waterfall",
    "id": 114
}, {
    "color": [112, 224, 255],
    "isthing": 0,
    "name": "tent",
    "id": 115
}, {
    "color": [70, 184, 160],
    "isthing": 1,
    "name": "bag",
    "id": 116
}, {
    "color": [163, 0, 255],
    "isthing": 1,
    "name": "minibike",
    "id": 117
}, {
    "color": [153, 0, 255],
    "isthing": 0,
    "name": "cradle",
    "id": 118
}, {
    "color": [71, 255, 0],
    "isthing": 1,
    "name": "oven",
    "id": 119
}, {
    "color": [255, 0, 163],
    "isthing": 1,
    "name": "ball",
    "id": 120
}, {
    "color": [255, 204, 0],
    "isthing": 1,
    "name": "food",
    "id": 121
}, {
    "color": [255, 0, 143],
    "isthing": 1,
    "name": "step",
    "id": 122
}, {
    "color": [0, 255, 235],
    "isthing": 0,
    "name": "tank",
    "id": 123
}, {
    "color": [133, 255, 0],
    "isthing": 1,
    "name": "trade-name",
    "id": 124
}, {
    "color": [255, 0, 235],
    "isthing": 1,
    "name": "microwave",
    "id": 125
}, {
    "color": [245, 0, 255],
    "isthing": 1,
    "name": "pot",
    "id": 126
}, {
    "color": [255, 0, 122],
    "isthing": 1,
    "name": "animal",
    "id": 127
}, {
    "color": [255, 245, 0],
    "isthing": 1,
    "name": "bicycle",
    "id": 128
}, {
    "color": [10, 190, 212],
    "isthing": 0,
    "name": "lake",
    "id": 129
}, {
    "color": [214, 255, 0],
    "isthing": 1,
    "name": "dishwasher",
    "id": 130
}, {
    "color": [0, 204, 255],
    "isthing": 1,
    "name": "screen",
    "id": 131
}, {
    "color": [20, 0, 255],
    "isthing": 0,
    "name": "blanket",
    "id": 132
}, {
    "color": [255, 255, 0],
    "isthing": 1,
    "name": "sculpture",
    "id": 133
}, {
    "color": [0, 153, 255],
    "isthing": 1,
    "name": "hood",
    "id": 134
}, {
    "color": [0, 41, 255],
    "isthing": 1,
    "name": "sconce",
    "id": 135
}, {
    "color": [0, 255, 204],
    "isthing": 1,
    "name": "vase",
    "id": 136
}, {
    "color": [41, 0, 255],
    "isthing": 1,
    "name": "traffic-light",
    "id": 137
}, {
    "color": [41, 255, 0],
    "isthing": 1,
    "name": "tray",
    "id": 138
}, {
    "color": [173, 0, 255],
    "isthing": 1,
    "name": "ashcan",
    "id": 139
}, {
    "color": [0, 245, 255],
    "isthing": 1,
    "name": "fan",
    "id": 140
}, {
    "color": [71, 0, 255],
    "isthing": 0,
    "name": "pier",
    "id": 141
}, {
    "color": [122, 0, 255],
    "isthing": 0,
    "name": "crt-screen",
    "id": 142
}, {
    "color": [0, 255, 184],
    "isthing": 1,
    "name": "plate",
    "id": 143
}, {
    "color": [0, 92, 255],
    "isthing": 1,
    "name": "monitor",
    "id": 144
}, {
    "color": [184, 255, 0],
    "isthing": 1,
    "name": "bulletin-board",
    "id": 145
}, {
    "color": [0, 133, 255],
    "isthing": 0,
    "name": "shower",
    "id": 146
}, {
    "color": [255, 214, 0],
    "isthing": 1,
    "name": "radiator",
    "id": 147
}, {
    "color": [25, 194, 194],
    "isthing": 1,
    "name": "glass",
    "id": 148
}, {
    "color": [102, 255, 0],
    "isthing": 1,
    "name": "clock",
    "id": 149
}, {
    "color": [92, 0, 255],
    "isthing": 1,
    "name": "flag",
    "id": 150
}]


def get_ade20k_meta() -> Sequence[Any]:
  return copy.deepcopy(_ADE20K_META)


def get_ade20k_class_has_instances_list() -> Sequence[int]:
  return tuple([x["id"] for x in _ADE20K_META if x["isthing"] == 1])


def get_id_mapping_inverse() -> Sequence[int]:
  id_mapping_inverse = (255,) + tuple(range(150))
  return id_mapping_inverse

