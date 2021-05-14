# Panoptic-DeepLab

Panoptic-DeepLab is a state-of-the-art **box-free** system for panoptic
segmentation, where the goal is to assign a unique value, encoding both
semantic label (e.g., person, car) and instance ID (e.g., instance_1,
instance_2), to every pixel in an image.

Panoptic-DeepLab improves over the DeeperLab model (one of the first box-free
systems for panoptic segmentation, which combines DeepLabv3+ and PersonLab) by
simplifying the class-agnostic instance detection to only use a center keypoint.
As a result, Panoptic-DeepLab predicts three outputs: (1) semantic segmentation,
(2) instance center heatmap, and (3) instance center regression.

The class-agnostic instance segmentation is first obtained by grouping
the predicted foreground pixels (inferred by semantic segmentation) to their
closest predicted instance centers. To generate final panoptic segmentation, we
then fuse the class-agnostic instance segmentation with semantic segmentation by
the efficient majority-vote scheme.


<p align="center">
   <img src="../img/panoptic_deeplab.png" width=800>
</p>


## Prerequisite

1. Make sure the software is properly [installed](../setup/installation.md).

2. Make sure the target dataset is correctly prepared (e.g.,
[Cityscapes](../setup/cityscapes.md)).

3. Download the ImageNet pretrained
[checkpoints](./imagenet_pretrained_checkpoints.md), and update the
`initial_checkpoint` path in the config files.

## Model Zoo

### Cityscapes Panoptic Segmentation

We provide checkpoints pretrained on Cityscapes train-fine set below. If you
would like to train those models by yourself, please find the corresponding
config files under the directory
[configs/cityscapes/panoptic_deeplab](../../configs/cityscapes/panoptic_deeplab).

Backbone | Output stride | Output resolution | PQ&dagger; | AP<sup>Mask</sup>&dagger; | mIoU
-------- | :-----------: | :---------------: | :---: | :---: | :---:
ResNet-50 ([config](../../configs/cityscapes/panoptic_deeplab/resnet50_os32_merge_with_pure_tf_func.textproto), [ckpt](http://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_os32_panoptic_deeplab_cityscapes_trainfine.tar.gz)) | 32 | 1024 x 2048 | 60.24 | 30.01 | 76.36

&dagger;: See Q4 in [FAQ](../faq.md).


## Citing Panoptic-DeepLab

If you find this code helpful in your research or wish to refer to the baseline
results, please use the following BibTeX entry.

* Panoptic-DeepLab:

```
@inproceedings{panoptic_deeplab_2020,
  author={Bowen Cheng and Maxwell D Collins and Yukun Zhu and Ting Liu and Thomas S Huang and Hartwig Adam and Liang-Chieh Chen},
  title={{Panoptic-DeepLab}: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation},
  booktitle={CVPR},
  year={2020}
}

```

If you use the Wide-ResNet-41 backbone, please consider citing

* Naive-Student:

```
@inproceedings{naive_student_2020,
  title={{Naive-Student: Leveraging Semi-Supervised Learning in Video Sequences for Urban Scene Segmentation}},
  author={Chen, Liang-Chieh and Lopes, Raphael Gontijo and Cheng, Bowen and Collins, Maxwell D and Cubuk, Ekin D and Zoph, Barret and Adam, Hartwig and Shlens, Jonathon},
  booktitle={ECCV},
  year={2020}
}
```

If you use the SWideRNet backbone w/ Switchable Atrous Convolution,
please consider citing

* SWideRNet:

```
@article{swidernet_2020,
  title={Scaling Wide Residual Networks for Panoptic Segmentation},
  author={Chen, Liang-Chieh and Wang, Huiyu and Qiao, Siyuan},
  journal={arXiv:2011.11675},
  year={2020}
}

```

* Swichable Atrous Convolution (SAC):

```
@inproceedings{detectors_2021,
  title={{DetectoRS}: Detecting Objects with Recursive Feature Pyramid and Switchable Atrous Convolution},
  author={Qiao, Siyuan and Chen, Liang-Chieh and Yuille, Alan},
  booktitle={CVPR},
  year={2021}
}

```

If you use the MobileNetv3 backbone, please consider citing

* MobileNetv3

```
@inproceedings{howard2019searching,
  title={Searching for {MobileNetV3}},
  author={Howard, Andrew and Sandler, Mark and Chu, Grace and Chen, Liang-Chieh and Chen, Bo and Tan, Mingxing and Wang, Weijun and Zhu, Yukun and Pang, Ruoming and Vasudevan, Vijay and others},
  booktitle={ICCV},
  year={2019}
}
```

### References

1. Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr
   Dollar. "Panoptic segmentation." In CVPR, 2019.

2. Alex Kendall, Yarin Gal, and Roberto Cipolla. "Multi-task learning using
   uncertainty to weigh losses for scene geometry and semantics." In CVPR, 2018.

3. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual
   learning for image recognition." In CVPR, 2016.

4. Sergey Zagoruyko and Nikos Komodakis. "Wide residual networks." In BMVC,
   2016.

5. Zifeng Wu, Chunhua Shen, and Anton Van Den Hengel. "Wider or deeper:
   Revisiting the ResNet model for visual recognition." Pattern Recognition,
   2019.
