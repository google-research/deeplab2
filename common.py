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

"""This file contains common methods and constants used across this framework."""

# Prediction keys used by the model output dictionary.
PRED_PANOPTIC_KEY = 'panoptic_pred'
PRED_SEMANTIC_KEY = 'semantic_pred'
PRED_INSTANCE_KEY = 'instance_pred'
PRED_INSTANCE_CENTER_KEY = 'instance_center_pred'


PRED_SEMANTIC_LOGITS_KEY = 'semantic_logits'
PRED_SEMANTIC_PROBS_KEY = 'semantic_probs'
PRED_SEMANTIC_SCORES_KEY = 'semantic_scores'
PRED_INSTANCE_SCORES_KEY = 'instance_scores'
PRED_CENTER_HEATMAP_KEY = 'center_heatmap'
PRED_OFFSET_MAP_KEY = 'offset_map'
PRED_FRAME_OFFSET_MAP_KEY = 'frame_offset_map'
PRED_NEXT_OFFSET_MAP_KEY = 'next_offset_map'
PRED_NEXT_PANOPTIC_KEY = 'next_panoptic_pred'
PRED_CONCAT_NEXT_PANOPTIC_KEY = 'concat_next_panoptic_pred'
PRED_DEPTH_KEY = 'depth_pred'

PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY = 'pixel_space_normalized_feature'
PRED_PIXEL_SPACE_MASK_LOGITS_KEY = 'pixel_space_mask_logits'
PRED_TRANSFORMER_CLASS_LOGITS_KEY = 'transformer_class_logits'

PRED_AUXILIARY_OUTPUTS = 'auxiliary_outputs_pred'

# Ground-truth keys used by the model.
GT_PANOPTIC_KEY = 'panoptic_gt'
GT_SEMANTIC_KEY = 'semantic_gt'
GT_INSTANCE_CENTER_KEY = 'instance_center_gt'
GT_INSTANCE_REGRESSION_KEY = 'instance_regression_gt'
GT_FRAME_OFFSET_KEY = 'frame_offset_gt'
GT_IS_CROWD = 'is_crowd_gt'
GT_THING_ID_MASK_KEY = 'thing_id_mask_gt'
GT_THING_ID_CLASS_KEY = 'thing_id_class_gt'
GT_NEXT_INSTANCE_REGRESSION_KEY = 'next_instance_regression_gt'
GT_DEPTH_KEY = 'depth_gt'

# Raw labels.
GT_PANOPTIC_RAW = 'panoptic_raw'
GT_SEMANTIC_RAW = 'semantic_raw'
GT_IS_CROWD_RAW = 'is_crowd_raw'
GT_SIZE_RAW = 'size_raw'
GT_NEXT_PANOPTIC_RAW = 'next_panoptic_raw'
GT_DEPTH_RAW = 'depth_raw'

# Loss keys.
SEMANTIC_LOSS = 'semantic_loss'
CENTER_LOSS = 'center_loss'
REGRESSION_LOSS = 'regression_loss'
MOTION_LOSS = 'motion_loss'
NEXT_REGRESSION_LOSS = 'next_regression_loss'
DEPTH_LOSS = 'depth_loss'
PQ_STYLE_LOSS = 'pq_style_loss'
# The PQ-style loss consists of a class term and a mask dice term.
PQ_STYLE_LOSS_CLASS_TERM = 'pq_style_loss_class_term'
PQ_STYLE_LOSS_MASK_DICE_TERM = 'pq_style_loss_mask_dice_term'
MASK_ID_CROSS_ENTROPY_LOSS = 'mask_id_cross_entropy_loss'
INSTANCE_DISCRIMINATION_LOSS = 'instance_discrimination_loss'
TOTAL_LOSS = 'total_loss'

# Weight keys used by the model.
SEMANTIC_LOSS_WEIGHT_KEY = 'semantic_loss_weight'
CENTER_LOSS_WEIGHT_KEY = 'center_loss_weight'
REGRESSION_LOSS_WEIGHT_KEY = 'regression_loss_weight'
FRAME_REGRESSION_LOSS_WEIGHT_KEY = 'frame_regression_loss_weight'
NEXT_REGRESSION_LOSS_WEIGHT_KEY = 'next_regression_loss_weight'
DEPTH_LOSS_WEIGHT_KEY = 'depth_loss_weight'

# Misc.
RESIZED_IMAGE = 'resized_image'
IMAGE = 'image'
IMAGE_NAME = 'image_name'
# For PVPS, we use sequence_id + frame_id to uniquely identify a frame.
SEQUENCE_ID = 'sequence_id'
FRAME_ID = 'frame_id'
NEXT_IMAGE = 'next_image'

# TfExample keys.
KEY_ENCODED_IMAGE = 'image/encoded'
KEY_ENCODED_PREV_IMAGE = 'prev_image/encoded'
KEY_ENCODED_NEXT_IMAGE = 'next_image/encoded'
KEY_IMAGE_FILENAME = 'image/filename'
KEY_IMAGE_FORMAT = 'image/format'
KEY_IMAGE_HEIGHT = 'image/height'
KEY_IMAGE_WIDTH = 'image/width'
KEY_IMAGE_CHANNELS = 'image/channels'
KEY_ENCODED_LABEL = 'image/segmentation/class/encoded'
KEY_ENCODED_PREV_LABEL = 'prev_image/segmentation/class/encoded'
KEY_ENCODED_NEXT_LABEL = 'next_image/segmentation/class/encoded'
KEY_LABEL_FORMAT = 'image/segmentation/class/format'
KEY_SEQUENCE_ID = 'video/sequence_id'
KEY_FRAME_ID = 'video/frame_id'
KEY_ENCODED_DEPTH = 'image/depth/encoded'
KEY_DEPTH_FORMAT = 'image/depth/format'

# TfExample keys for the panoramic setting, where individual camera keys are
# obtained by formatting with the camera name.
KEY_PER_CAMERA_ENCODED_IMAGE = 'image/encoded/%s'
KEY_PER_CAMERA_ENCODED_NEXT_IMAGE = 'next_image/encoded/%s'
KEY_PER_CAMERA_IMAGE_HEIGHT = 'image/height/%s'
KEY_PER_CAMERA_IMAGE_WIDTH = 'image/width/%s'
KEY_PER_CAMERA_ENCODED_LABEL = 'image/segmentation/class/encoded/%s'
KEY_PER_CAMERA_ENCODED_NEXT_LABEL = 'next_image/segmentation/class/encoded/%s'
KEY_PER_CAMERA_ENCODED_DEPTH = 'image/depth/encoded/%s'

# Checkpoint Items
# All models
CKPT_SEMANTIC_LAST_LAYER = 'semantic_last_layer'

# DeepLabV3
CKPT_DEEPLABV3_ASPP = 'deeplab_v3_aspp'
CKPT_DEEPLABV3_CLASSIFIER_CONV_BN_ACT = 'classifier_conv_bn_act'

# DeepLabV3+
CKPT_DEEPLABV3PLUS_ASPP = 'deeplab_v3plus_aspp'
CKPT_DEEPLABV3PLUS_PROJECT_CONV_BN_ACT = 'deeplab_v3plus_project_conv_bn_act'
CKPT_DEEPLABV3PLUS_FUSE = 'deeplab_v3plus_fuse'

# Panoptic-DeepLab
CKPT_SEMANTIC_DECODER = 'semantic_decoder'
CKPT_SEMANTIC_HEAD_WITHOUT_LAST_LAYER = 'semantic_head_without_last_layer'

CKPT_INSTANCE_DECODER = 'instance_decoder'
CKPT_INSTANCE_CENTER_HEAD_WITHOUT_LAST_LAYER = ('instance_center_head'
                                                '_without_last_layer')
CKPT_INSTANCE_CENTER_HEAD_LAST_LAYER = 'instance_center_head_last_layer'
CKPT_INSTANCE_REGRESSION_HEAD_WITHOUT_LAST_LAYER = ('instance_regression_head'
                                                    '_without_last_layer')
CKPT_INSTANCE_REGRESSION_HEAD_LAST_LAYER = 'instance_regression_head_last_layer'

# Motion-DeepLab
CKPT_MOTION_REGRESSION_HEAD_WITHOUT_LAST_LAYER = ('motion_regression_head'
                                                  '_without_last_layer')
CKPT_MOTION_REGRESSION_HEAD_LAST_LAYER = 'motion_regression_head_last_layer'

# ViP-DeepLab
CKPT_NEXT_INSTANCE_DECODER = 'next_instance_decoder'
CKPT_NEXT_INSTANCE_REGRESSION_HEAD_WITHOUT_LAST_LAYER = (
    'next_instance_regression_head_without_last_layer')
CKPT_NEXT_INSTANCE_REGRESSION_HEAD_LAST_LAYER = (
    'next_instance_regression_head_last_layer')
CKPT_DEPTH_HEAD_WITHOUT_LAST_LAYER = 'depth_head_without_last_layer'
CKPT_DEPTH_HEAD_LAST_LAYER = 'depth_head_last_layer'

# MaX-DeepLab
CKPT_PIXEL_SPACE_HEAD = 'pixel_space_head'
CKPT_TRANSFORMER_MASK_HEAD = 'transformer_mask_head'
CKPT_TRANSFORMER_CLASS_HEAD = 'transformer_class_head'
CKPT_PIXEL_SPACE_FEATURE_BATCH_NORM = 'pixel_space_feature_batch_norm'
CKPT_PIXEL_SPACE_MASK_BATCH_NORM = 'pixel_space_mask_batch_norm'

# Supported Tasks
TASK_PANOPTIC_SEGMENTATION = 'panoptic_segmentation'
TASK_INSTANCE_SEGMENTATION = 'instance_segmentation'
TASK_VIDEO_PANOPTIC_SEGMENTATION = 'video_panoptic_segmentation'
TASK_DEPTH_AWARE_VIDEO_PANOPTIC_SEGMENTATION = (
    'depth_aware_video_panoptic_segmentation')
