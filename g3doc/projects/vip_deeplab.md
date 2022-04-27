# ViP-DeepLab

ViP-DeepLab is a state-of-the-art system for the task depth-aware video panoptic
segmentation that aims to simultaneously tackle video panoptic segmentation [1]
and monocular depth estimation [2]. The goal is to assign a unique value that
encodes both semantic label and temporally consistent instance ID to every pixel
in an image while estimating its depth from the camera.

ViP-DeepLab extends Panoptic-DeepLab by adding network branches for depth and
video predictions. It is a unified model that jointly performs video panoptic
segmentation and monocular depth estimation for each pixel on the image plane,
and achieves state-of-the-art performance on several academic datasets for the
sub-tasks. The GIF below shows the results of ViP-DeepLab.

<p align="center">
   <img src="../img/vip_deeplab/demo.gif" width=800>
</p>

## Prerequisite

1.  Make sure the software is properly [installed](../setup/installation.md).

2.  Make sure the target dataset is correctly prepared (e.g.,
    [SemKITTI-DVPS](https://github.com/joe-siyuan-qiao/ViP-DeepLab/tree/master/semkitti-dvps),
    [Cityscapes-DVPS](https://github.com/joe-siyuan-qiao/ViP-DeepLab/tree/master/cityscapes-dvps)).

3.  Download the pretrained Panoptic-DeepLab
    [checkpoints](./panoptic_deeplab.md) with ResNet-50-Beta for Cityscapes
    panoptic segmentation, and update the `initial_checkpoint` path in the
    config files.

## Model Zoo

In the Model Zoo, we present several checkpoints trained on Cityscapes-DVPS and
SemKITTI-DVPS. We report the results obtained by *single-scale* inference and
*Cityscapes* pretrained checkpoints for model initialization. The numbers are
obtained from running the code in *eval* mode. Note that the numbers here are
different from the results reported in the paper due to the differences in the
TF-2 re-implementation (the original paper used TF-1) and initial checkpoints,
e.g., the open-source model does not have Mapillary-Vistas pretraining,
cascade-ASPP, recursive feature pyramid (RFP), and Cityscapes video pseudo-label
pretraining.

Dataset         | Backbone                                                                                                                                                                                                                              | Output Stride | VPQ-1 | VPQ-2 | VPQ-3 | VPQ-4 | VPQ   | AbsRel
--------------- | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :---: | :---: | :---: | :---: | :---: | :----:
Cityscapes-DVPS | ResNet-50-Beta ([config](../../configs/cityscapes_dvps/vip_deeplab/resnet50_beta_os32.textproto), [ckpt](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_beta_os32_vip_deeplab_cityscapes_dvps_train.tar.gz)) | 32            | 60.61 | 53.14 | 49.89 | 47.66 | 52.82 | 0.112
Cityscapes-DVPS | WideResNet-41 ([config](../../configs/cityscapes_dvps/vip_deeplab/wide_resnet41_os16.textproto), [ckpt](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/wide_resnet41_os16_vip_deeplab_cityscapes_dvps_train.tar.gz))  | 16            | 63.24 | 55.56 | 52.10 | 49.85 | 55.19 | 0.114

Dataset       | Backbone                                                                                                                                                                                                                          | Output Stride | VPQ-1 | VPQ-5 | VPQ-10 | VPQ-20 | VPQ   | AbsRel
------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------: | :---: | :---: | :----: | :----: | :---: | :----:
SemKITTI-DVPS | ResNet-50-Beta ([config](../../configs/semkitti_dvps/vip_deeplab/resnet50_beta_os32.textproto), [ckpt](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_beta_os32_vip_deeplab_semkitti_dvps_train.tar.gz)) | 32            | 48.6  | 45.8  | 44.5   | 43.6   | 45.63 | 0.139

## Citing ViP-DeepLab

If you find this code helpful in your research or wish to refer to the baseline
results, please use the following BibTeX entry.

*   ViP-DeepLab:

```
@inproceedings{vip_deeplab_2021,
  author={Siyuan Qiao and Yukun Zhu and Hartwig Adam and Alan Yuille and Liang-Chieh Chen},
  title={{ViP-DeepLab}: Learning Visual Perception with Depth-aware Video Panoptic Segmentation},
  booktitle={CVPR},
  year={2021}
}

```

*   Panoptic-DeepLab:

```
@inproceedings{panoptic_deeplab_2020,
  author={Bowen Cheng and Maxwell D Collins and Yukun Zhu and Ting Liu and Thomas S Huang and Hartwig Adam and Liang-Chieh Chen},
  title={{Panoptic-DeepLab}: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation},
  booktitle={CVPR},
  year={2020}
}

```

### References

1.  Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. "Video Panoptic
    Segmentation." In CVPR, 2020.

2.  David Eigen, Christian Puhrsch, and Rob Fergus. "Depth Map Prediction from a
    Single Image Using a Multi-Scale Deep Network." In NeurIPS, 2014.

3.  Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus
    Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele,
    "The Cityscapes Dataset for Semantic Urban Scene Understanding." In
    CVPR, 2016.

4.  Andreas Geiger and Philip Lenz and Raquel Urtasun, "Are We Ready for
    Autonomous Driving? The KITTI Vision Benchmark Suite." In CVPR, 2012.

5.  Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke,
    Cyrill Stachniss, and Jurgen Gall. "SemanticKITTI: A Dataset for Semantic
    Scene Understanding of LiDAR Sequences." In ICCV, 2019.
