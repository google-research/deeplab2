# coding=utf-8
# Copyright 2022 The Deeplab2 Authors.
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

"""Provides data from segmentation datasets.

Currently, we support the following datasets:

1. Cityscapes dataset (https://www.cityscapes-dataset.com).

The Cityscapes dataset contains 19 semantic labels (such as road, person, car,
and so on) for urban street scenes.


2. KITTI-STEP (http://www.cvlibs.net/datasets/kitti/).

The KITTI-STEP enriches the KITTI-MOTS data with additional `stuff'
anntotations.

3. MOTChallenge-STEP (https://motchallenge.net/).

The MOTChallenge-STEP enriches the MOTSChallenge data with additional `stuff'
annotations.

4. MSCOCO panoptic segmentation (http://cocodataset.org/#panoptic-2018).

Panoptic segmentation annotations for MSCOCO dataset. Note that we convert the
provided MSCOCO panoptic segmentation format to the following one:
panoptic label = semantic label * 256 + instance id.

5. Cityscapes-DVPS (https://github.com/joe-siyuan-qiao/ViP-DeepLab)

The Cityscapes-DVPS dataset augments Cityscapes-VPS
(https://github.com/mcahny/vps) with depth annotations.

6. SemKITTI-DVPS (https://github.com/joe-siyuan-qiao/ViP-DeepLab)

The SemKITTI-DVPS dataset converts 3D point annotations of SemanticKITTI
(http://www.semantic-kitti.org) into 2D image labels.

References:

- Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus
  Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele, "The
  Cityscapes Dataset for Semantic Urban Scene Understanding." In CVPR, 2016.

- Andreas Geiger and Philip Lenz and Raquel Urtasun, "Are we ready for
  Autonomous Driving? The KITTI Vision Benchmark Suite." In CVPR, 2012.

- Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr
  Dollar, "Panoptic Segmentation." In CVPR, 2019.

- Tsung-Yi Lin, Michael Maire, Serge J. Belongie, Lubomir D. Bourdev, Ross B.
  Girshick, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C.
  Lawrence Zitnick, "Microsoft COCO: common objects in context." In ECCV, 2014.

- Anton Milan, Laura Leal-Taixe, Ian Reid, Stefan Roth, and Konrad Schindler,
  "Mot16: A benchmark for multi-object tracking." arXiv:1603.00831, 2016.

- Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin
  Balachandar Gnana Sekar, Andreas Geiger, and Bastian Leibe. "MOTS:
  Multi-object tracking and segmentation." In CVPR, 2019

- Mark Weber, Jun Xie, Maxwell Collins, Yukun Zhu, Paul Voigtlaender, Hartwig
  Adam, Bradley Green, Andreas Geiger, Bastian Leibe, Daniel Cremers, Aljosa
  Osep, Laura Leal-Taixe, and Liang-Chieh Chen, "STEP: Segmenting and Tracking
  Every Pixel." arXiv: 2102.11859, 2021.

- Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. "Video panoptic
  segmentation." In CVPR, 2020.

- Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke, Cyrill
  Stachniss, and Jurgen Gall. "Semantickitti: A dataset for semantic scene
  understanding of lidar sequences." In ICCV, 2019.

- Siyuan Qiao, Yukun Zhu, Hartwig Adam, Alan Yuille, and Liang-Chieh Chen.
  "ViP-DeepLab: Learning Visual Perception with Depth-aware Video Panoptic
  Segmentation." In CVPR, 2021.
"""

import collections


# Dataset names.
_CITYSCAPES_PANOPTIC = 'cityscapes_panoptic'
_KITTI_STEP = 'kitti_step'
_MOTCHALLENGE_STEP = 'motchallenge_step'
_CITYSCAPES_DVPS = 'cityscapes_dvps'
_SEMKITTI_DVPS = 'semkitti_dvps'
_COCO_PANOPTIC = 'coco_panoptic'

# Colormap names.
CITYSCAPES_COLORMAP = 'cityscapes'
MOTCHALLENGE_COLORMAP = 'motchallenge'
COCO_COLORMAP = 'coco'


# Named tuple to describe dataset properties.
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor', [
        'dataset_name',  # Dataset name.
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',   # Number of semantic classes.
        'ignore_label',  # Ignore label value used for semantic segmentation.

        # Fields below are used for panoptic segmentation and will be None for
        # Semantic segmentation datasets.
        # Label divisor only used in panoptic segmentation annotation to infer
        # semantic label and instance id.
        'panoptic_label_divisor',
        # A tuple of classes that contains instance annotations. For example,
        # 'person' class has instance annotations while 'sky' does not.
        'class_has_instances_list',
        # A flag indicating whether the dataset is a video dataset that contains
        # sequence IDs and frame IDs.
        'is_video_dataset',
        # A string specifying the colormap that should be used for
        # visualization. E.g. 'cityscapes'.
        'colormap',
        # A flag indicating whether the dataset contains depth annotation.
        'is_depth_dataset',
        # The ignore label for depth annotations.
        'ignore_depth',
    ]
)

CITYSCAPES_PANOPTIC_INFORMATION = DatasetDescriptor(
    dataset_name=_CITYSCAPES_PANOPTIC,
    splits_to_sizes={'train_fine': 2975,
                     'val_fine': 500,
                     'trainval_fine': 3475,
                     'test_fine': 1525},
    num_classes=19,
    ignore_label=255,
    panoptic_label_divisor=1000,
    class_has_instances_list=tuple(range(11, 19)),
    is_video_dataset=False,
    colormap=CITYSCAPES_COLORMAP,
    is_depth_dataset=False,
    ignore_depth=None,
)

KITTI_STEP_INFORMATION = DatasetDescriptor(
    dataset_name=_KITTI_STEP,
    splits_to_sizes={'train': 5027,
                     'val': 2981,
                     'test': 11095},
    num_classes=19,
    ignore_label=255,
    panoptic_label_divisor=1000,
    class_has_instances_list=(11, 13),
    is_video_dataset=True,
    colormap=CITYSCAPES_COLORMAP,
    is_depth_dataset=False,
    ignore_depth=None,
)

MOTCHALLENGE_STEP_INFORMATION = DatasetDescriptor(
    dataset_name=_MOTCHALLENGE_STEP,
    splits_to_sizes={'train': 525,  # Sequence 9.
                     'val': 600,  # Sequence 2.
                     'test': 0},
    num_classes=7,
    ignore_label=255,
    panoptic_label_divisor=1000,
    class_has_instances_list=(4,),
    is_video_dataset=True,
    colormap=MOTCHALLENGE_COLORMAP,
    is_depth_dataset=False,
    ignore_depth=None,
)

CITYSCAPES_DVPS_INFORMATION = DatasetDescriptor(
    dataset_name=_CITYSCAPES_DVPS,
    # The numbers of images are 2400/300/300 for train/val/test. Here, the
    # sizes are the number of consecutive frame pairs. As each sequence has 6
    # frames, the number of pairs for the train split is 2400 / 6 * 5 = 2000.
    # Similarly, we get 250 pairs for the val split and the test split.
    splits_to_sizes={'train': 2000,
                     'val': 250,
                     'test': 250},
    num_classes=19,
    ignore_label=32,
    panoptic_label_divisor=1000,
    class_has_instances_list=tuple(range(11, 19)),
    is_video_dataset=True,
    colormap=CITYSCAPES_COLORMAP,
    is_depth_dataset=True,
    ignore_depth=0,
)

SEMKITTI_DVPS_INFORMATION = DatasetDescriptor(
    dataset_name=_SEMKITTI_DVPS,
    splits_to_sizes={'train': 19120,
                     'val': 4070,
                     'test': 4340},
    num_classes=19,
    ignore_label=255,
    panoptic_label_divisor=65536,
    class_has_instances_list=tuple(range(8)),
    is_video_dataset=True,
    # Reuses Cityscapes colormap.
    colormap=CITYSCAPES_COLORMAP,
    is_depth_dataset=True,
    ignore_depth=0,
)

COCO_PANOPTIC_INFORMATION = DatasetDescriptor(
    dataset_name=_COCO_PANOPTIC,
    splits_to_sizes={'train': 118287,
                     'val': 5000,
                     'test': 40670},
    num_classes=134,
    ignore_label=0,
    panoptic_label_divisor=256,
    class_has_instances_list=tuple(range(1, 81)),
    is_video_dataset=False,
    colormap=COCO_COLORMAP,
    is_depth_dataset=False,
    ignore_depth=None,
)

MAP_NAME_TO_DATASET_INFO = {
    _CITYSCAPES_PANOPTIC: CITYSCAPES_PANOPTIC_INFORMATION,
    _KITTI_STEP: KITTI_STEP_INFORMATION,
    _MOTCHALLENGE_STEP: MOTCHALLENGE_STEP_INFORMATION,
    _CITYSCAPES_DVPS: CITYSCAPES_DVPS_INFORMATION,
    _COCO_PANOPTIC: COCO_PANOPTIC_INFORMATION,
    _SEMKITTI_DVPS: SEMKITTI_DVPS_INFORMATION,
}

MAP_NAMES = list(MAP_NAME_TO_DATASET_INFO.keys())
