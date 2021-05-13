References below are really meant for reference when writing the doc.
Please remove the references once ready.

References:

* https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

# Axial-DeepLab

Cool model.

## Prerequisite

1. Make sure the software is properly [installed](../setup/installation.md).

2. Make sure the target dataset is correctly prepared (e.g.,
[Cityscapes](../setup/cityscapes.md)).

3. Download the ImageNet pretrained
[checkpoints](./imagenet_pretrained_checkpoints.md), and update the
`initial_checkpoint` path in the config files.

## Model Zoo

#### Cityscapes Panoptic Segmentation

We provide checkpoints pretrained on Cityscapes train-fine set below. If you
would like to train those models by yourself, please find the corresponding
config files under this [directory](../../configs/cityscapes/axial_deeplab).

Backbone | Output stride | Output resolution | PQ&dagger; | AP<sup>Mask</sup>&dagger; | mIoU | Checkpoint
-------- | :-----------: | :---------------: | :---: | :---: | :---: | :---:

&dagger;: See Q4 in [FAQ](../faq.md).


## Citing Axial-DeepLab

If you find this code helpful in your research or wish to refer to the baseline
results, please use the following BibTeX entry.

* Axial-DeepLab:

```
@inproceedings{axial_deeplab_2020,
  author={Huiyu Wang and Yukun Zhu and Bradley Green and Hartwig Adam and Alan Yuille and Liang-Chieh Chen},
  title={{Axial-DeepLab}: Stand-Alone Axial-Attention for Panoptic Segmentation},
  booktitle={ECCV},
  year={2020}
}

```

* Panoptic-DeepLab:

```
@inproceedings{panoptic_deeplab_2020,
  author={Bowen Cheng and Maxwell D Collins and Yukun Zhu and Ting Liu and Thomas S Huang and Hartwig Adam and Liang-Chieh Chen},
  title={{Panoptic-DeepLab}: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation},
  booktitle={CVPR},
  year={2020}
}

```

If you use the SWideRNet backbone w/ axial attention, please consider
citing

* SWideRNet:

```
@article{swidernet_2020,
  title={Scaling Wide Residual Networks for Panoptic Segmentation},
  author={Chen, Liang-Chieh and Wang, Huiyu and Qiao, Siyuan},
  journal={arXiv:2011.11675},
  year={2020}
}

```

If you use the MaX-DeepLab-{S,L} backbone, please consider
citing

* MaX-DeepLab:

```
@inproceedings{max_deeplab_2021,
  author={Huiyu Wang and Yukun Zhu and Hartwig Adam and Alan Yuille and Liang-Chieh Chen},
  title={{MaX-DeepLab}: End-to-End Panoptic Segmentation with Mask Transformers},
  booktitle={CVPR},
  year={2021}
}

```
