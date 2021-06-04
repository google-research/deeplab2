References below are really meant for reference when writing the doc.
Please remove the references once ready.

References:

* https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
* https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

# MaX-DeepLab

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
config files under this [directory](../../configs/cityscapes/max_deeplab).

All the reported results are obtained by a *single-scale* inference.

Backbone | Output stride | Output resolution | PQ&dagger; | AP<sup>Mask</sup>&dagger; | mIoU | Checkpoint
-------- | :-----------: | :---------------: | :---: | :---: | :---: | :---:


## Citing MaX-DeepLab

If you find this code helpful in your research or wish to refer to the baseline
results, please use the following BibTeX entry.

* MaX-DeepLab:

```
@inproceedings{max_deeplab_2021,
  author={Huiyu Wang and Yukun Zhu and Hartwig Adam and Alan Yuille and Liang-Chieh Chen},
  title={{MaX-DeepLab}: End-to-End Panoptic Segmentation with Mask Transformers},
  booktitle={CVPR},
  year={2021}
}

```
