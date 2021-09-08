# DeepLab2: A TensorFlow Library for Deep Labeling

DeepLab2 is a TensorFlow library for deep labeling, aiming to provide a
unified and state-of-the-art TensorFlow codebase for dense pixel labeling tasks,
including, but not limited to semantic segmentation, instance segmentation,
panoptic segmentation, depth estimation, or even video panoptic segmentation.

Deep labeling refers to solving computer vision problems by assigning a
predicted value for each pixel in an image with a deep neural network. As
long as the problem of interest could be formulated in this way, DeepLab2
should serve the purpose. Additionally, this codebase includes our recent and
state-of-the-art research models on deep labeling. We hope you will find it
useful for your projects.

## Change logs

* 09/07/2021: Add numpy implementation of Segmentation and Tracking Quality.
Find it [here](evaluation/numpy/segmentation_and_tracking_quality.py)

* 09/06/2021: Update Panoptic-DeepLab w/ MobileNetv3 backbone results on
  Cityscapes.

* 08/13/2021: Open-source MaX-DeepLab-L COCO checkpoints (51.3% PQ on COCO
  val set).

* 07/26/2021: Add ViP-DeepLab support for SemKITTI-DVPS.

* 07/07/2021: KITTI-STEP and MOTChallenge-STEP are ready to use.

* 06/07/2021: Add hungarian matching support on TPU for MaX-DeepLab, thanks
  to the help from Jiquan Ngiam and Amil Merchant.

* 06/01/2021: "Hello, World!", DeepLab2 made publicly available.

## Installation

See [Installation](g3doc/setup/installation.md).

## Dataset preparation

The dataset needs to be converted to TFRecord. We provide some examples below.

* <a href='g3doc/setup/cityscapes.md'>Cityscapes</a><br>
* <a href='g3doc/setup/coco.md'>COCO</a><br>
* <a href='g3doc/setup/kitti_step.md'>KITTI-STEP</a><br>
* <a href='g3doc/setup/motchallenge_step.md'>MOTChallenge-STEP</a><br>

Some guidances about how to convert your own dataset.

* <a href='g3doc/setup/your_own_dataset.md'>Your Own Dataset</a><br>

## Projects

We list a few projects that use DeepLab2.

* <a href='g3doc/projects/panoptic_deeplab.md'>Panoptic-DeepLab</a><br>
* <a href='g3doc/projects/axial_deeplab.md'>Axial-DeepLab</a><br>
* <a href='g3doc/projects/max_deeplab.md'>MaX-DeepLab</a><br>
* <a href='g3doc/projects/motion_deeplab.md'>STEP (Motion-DeepLab)</a><br>
* <a href='g3doc/projects/vip_deeplab.md'>ViP-DeepLab</a><br>

## Colab Demo

*   <a href='https://colab.research.google.com/github/google-research/deeplab2/blob/main/DeepLab_Demo.ipynb'>Colab notebook for off-the-shelf inference.</a><br>

## Running DeepLab2

See [Getting Started](g3doc/setup/getting_started.md). In short, run the
following command:

To run DeepLab2 on GPUs, the following command should be used:

```bash
python trainer/train.py \
    --config_file=${CONFIG_FILE} \
    --mode={train | eval | train_and_eval | continuous_eval} \
    --model_dir=${BASE_MODEL_DIRECTORY} \
    --num_gpus=${NUM_GPUS}
```

## Contacts (Maintainers)

Please check <a href='g3doc/faq.md'>FAQ</a> if you have some questions before
reporting the issues.<br>

* Mark Weber, github: [markweberdev](https://github.com/markweberdev)
* Huiyu Wang, github: [csrhddlam](https://github.com/csrhddlam)
* Siyuan Qiao, github: [joe-siyuan-qiao](https://github.com/joe-siyuan-qiao)
* Jun Xie, github: [clairexie](https://github.com/clairexie)
* Maxwell D. Collins, github: [mcollinswisc](https://github.com/mcollinswisc)
* YuKun Zhu, github: [yknzhu](https://github.com/YknZhu)
* Liangzhe Yuan, github: [yuanliangzhe](https://github.com/yuanliangzhe)
* Dahun Kim, github: [mcahny](https://github.com/mcahny)
* Qihang Yu, github: [yucornetto](https://github.com/yucornetto)
* Liang-Chieh Chen, github: [aquariusjay](https://github.com/aquariusjay)

## Disclaimer

* Note that this library contains our **re-implemented** DeepLab models in
*TensorFlow2*, and thus may have some minor differences from the published
papers (e.g., learning rate).

* This is not an official Google product.

## Citing DeepLab2

If you find DeepLab2 useful for your project, please consider citing
`DeepLab2` along with the relevant DeepLab series.

* DeepLab2:

```
@article{deeplab2_2021,
  author={Mark Weber and Huiyu Wang and Siyuan Qiao and Jun Xie and Maxwell D. Collins and Yukun Zhu and Liangzhe Yuan and Dahun Kim and Qihang Yu and Daniel Cremers and Laura Leal-Taixe and Alan L. Yuille and Florian Schroff and Hartwig Adam and Liang-Chieh Chen},
  title={{DeepLab2: A TensorFlow Library for Deep Labeling}},
  journal={arXiv: 2106.09748},
  year={2021}
}

```

### References

1. Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus
   Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele.
   "The cityscapes dataset for semantic urban scene understanding." In CVPR,
   2016.

2. Andreas Geiger, Philip Lenz, and Raquel Urtasun. "Are we ready for
   autonomous driving? the kitti vision benchmark suite." In CVPR, 2012.

3. Jens Behley, Martin Garbade, Andres Milioto, Jan Quenzel, Sven Behnke,
   Cyrill Stachniss, and Jurgen Gall. "Semantickitti: A dataset for semantic
   scene understanding of lidar sequences." In ICCV, 2019.

4. Alexander Kirillov, Kaiming He, Ross Girshick, Carsten Rother, and Piotr
   Dollar. "Panoptic segmentation." In CVPR, 2019.

5. Dahun Kim, Sanghyun Woo, Joon-Young Lee, and In So Kweon. "Video panoptic
   segmentation." In CVPR, 2020.

6. Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona,
   Deva Ramanan, Piotr Dollar, and C Lawrence Zitnick. "Microsoft COCO:
   Common objects in context." In ECCV, 2014.

7. Patrick Dendorfer, Aljosa Osep, Anton Milan, Konrad Schindler, Daniel
   Cremers, Ian Reid, Stefan Roth, and Laura Leal-Taixe. "MOTChallenge: A
   Benchmark for Single-camera Multiple Target Tracking." IJCV, 2020.
