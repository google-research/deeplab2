# Motion-DeepLab

Motion-DeepLab is a unified model for the task of video panoptic segmentation,
which requires to segment and track every pixel. It is built on top of
Panoptic-DeepLab and uses an additional branch to regress each pixel to its
center location in the previous frame. Instead of using a single RGB image as
input, the network input contains two consecutive frames, i.e., the current and
previous frame, as well as the center heatmap from the previous frame, similar
to CenterTrack [1]. The output is used to assign consistent track IDs to all
instances throughout a video sequence.

## Prerequisite

1. Make sure the software is properly [installed](../setup/installation.md).

2. Make sure the target dataset is correctly prepared (e.g.,
[KITTI-STEP](../setup/kitti_step.md)).

3. Download the Cityscapes pretrained checkpoints listed below, and update
the `initial_checkpoint` path in the config files.

## Model Zoo

### KITTI-STEP Video Panoptic Segmentation

**Initial checkpoint**: We provide several Cityscapes pretrained checkpoints
for KITTI-STEP experiments. Please download them and update the
`initial_checkpoint` path in the config files.

Model | Download | Note |
-------- | :-----------: | :---------------: |
Panoptic-DeepLab | [initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine.tar.gz) | The initial checkpoint for single-frame baseline.
Motion-DeepLab | [initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine_netsurgery_first_layer.tar.gz) | The initial checkpoint for two-frame baseline.

We also provide checkpoints pretrained on KITTI-STEP below. If
you would like to train those models by yourself, please find the
corresponding config files under the directories
[configs/kitti/panoptic_deeplab (single-frame-baseline)](../../configs/kitti/panoptic_deeplab)
or
[configs/kitti/motion_deeplab (two-frame-baseline)](../../configs/kitti/motion_deeplab).

**Panoptic-DeepLab (single-frame-baseline)**:

Backbone                                                                                                                                                                                                                 | Output stride | Dataset split           | PQ&dagger; | AP<sup>Mask</sup>&dagger; | mIoU
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :-----------: | :---------------------: | :--------: | :-----------------------: | :--:
ResNet-50 ([config](../../configs/kitti/panoptic_deeplab/resnet50_os32.textproto), [ckpt](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_panoptic_deeplab_kitti_train.tar.gz))             | 32            | KITTI-STEP train set    | 48.31      | 42.22                     | 71.16
ResNet-50 ([config](../../configs/kitti/panoptic_deeplab/resnet50_os32_trainval.textproto), [ckpt](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_panoptic_deeplab_kitti_trainval.tar.gz)) | 32            | KITTI-STEP trainval set | -          | -                         | -

&dagger;: See Q4 in [FAQ](../faq.md).

This single-frame baseline could be used together with other state-of-the-art
optical flow methods (e.g., RAFT [2]) for propagating mask predictions
from one frame to another, as shown in our STEP paper.

**Motion-DeepLab (two-frame-baseline)**:

Backbone | Output stride | Dataset split | PQ&dagger; | AP<sup>Mask</sup>&dagger; | mIoU | STQ
-------- | :-----------: | :---------------: | :---: | :---: | :---: | :---:
ResNet-50 ([config](../../configs/kitti/motion_deeplab/resnet50_os32.textproto), [ckpt](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_motion_deeplab_kitti_train.tar.gz)) | 32 | KITTI-STEP train set | 42.08 | 37.52 | 63.15 | 57.7
ResNet-50 ([config](../../configs/kitti/motion_deeplab/resnet50_os32_trainval.textproto), [ckpt](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_motion_deeplab_kitti_trainval.tar.gz))| 32 | KITTI-STEP trainval set | - | - | - | -

&dagger;: See Q4 in [FAQ](../faq.md).

### MOTChallenge-STEP Video Panoptic Segmentation

**Initial checkpoint**: We provide several Cityscapes pretrained checkpoints
for MOTChallenge-STEP experiments. Please download them and update the
`initial_checkpoint` path in the config files.

Model | Download | Note |
-------- | :-----------: | :---------------: |
Panoptic-DeepLab | [initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine_netsurgery_last_layer.tar.gz) | The initial checkpoint for single-frame baseline.
Motion-DeepLab | [initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_panoptic_deeplab_cityscapes_crowd_trainfine_netsurgery_first_and_last_layer.tar.gz) | The initial checkpoint for two-frame baseline.

We also provide checkpoints pretrained on MOTChallenge-STEP below.
If you would like to train those models by yourself, please find the
corresponding config files under the directories for
[configs/motchallenge/panoptic_deeplab (single-frame-baseline)](../../configs/motchallenge/panoptic_deeplab)
or
[configs/motchallenge/motion_deeplab (two-frame-baseline)](../../configs/motchallenge/motion_deeplab).

**Panoptic-DeepLab (single-frame-baseline)**:

TODO: Add pretrained checkpoint.

Backbone | Output stride | Dataset split | PQ&dagger; | AP<sup>Mask</sup>&dagger; | mIoU
-------- | :-----------: | :---------------: | :---: | :---: | :---:
ResNet-50 ([config](../../configs/motchallenge/panoptic_deeplab/resnet50_os32.textproto)) | 32 | MOTChallenge-STEP train set | ? | ? | ?
ResNet-50 | 32 | MOTChallenge-STEP trainval set | - | - | -

&dagger;: See Q4 in [FAQ](../faq.md).

This single-frame baseline could be used together with other state-of-the-art
optical flow methods (e.g., RAFT [2]) for propagating mask predictions
from one frame to another, as shown in our STEP paper.

**Motion-DeepLab (two-frame-baseline)**:

TODO: Add pretrained checkpoint.

Backbone | Output stride | Dataset split | PQ&dagger; | AP<sup>Mask</sup>&dagger; | mIoU | STQ
-------- | :-----------: | :---------------: | :---: | :---: | :---: | :---:
ResNet-50 ([config](../../configs/motchallenge/motion_deeplab/resnet50_os32.textproto)) | 32 | MOTChallenge-STEP train set | ? | ? | ? |?
ResNet-50 | 32 | MOTChallenge-STEP trainval set | - | - | - | -

&dagger;: See Q4 in [FAQ](../faq.md).

## Citing Motion-DeepLab

If you find this code helpful in your research or wish to refer to the baseline
results, please use the following BibTeX entry.

* STEP (Motion-DeepLab):

```
@article{step_2021,
  author={Mark Weber and Jun Xie and Maxwell Collins and Yukun Zhu and Paul Voigtlaender and Hartwig Adam and Bradley Green and Andreas Geiger and Bastian Leibe and Daniel Cremers and Aljosa Osep and Laura Leal-Taixe and Liang-Chieh Chen},
  title={{STEP}: Segmenting and Tracking Every Pixel},
  journal={arXiv:2102.11859},
  year={2021}
}

```

### References

1. Xingyi Zhou, Vladlen Koltun, and Philipp Krahenbuhl. Tracking objects as
points. ECCV, 2020

2. Zachary Teed and Jia Deng. RAFT: recurrent all-pairs field transforms for
optical flow. In ECCV, 2020
