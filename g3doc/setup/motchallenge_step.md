# Run DeepLab2 on MOTChallenge-STEP dataset

## MOTChallenge-STEP dataset

MOTChallenge-STEP extends the existing [MOTChallenge](https://motchallenge.net/)
dataset with spatially and temporally dense annotations.

### Label Map

MOTChallenge-STEP dataset followings the same annotation and label policy as
[KITTI-STEP dataset](./kitti_step.md). Among the
[MOTChallenge](https://motchallenge.net/) dataset, 4 outdoor sequences are
annotated for MOTChallenge-STEP. In particular, these sequences are splitted
into 2 for training and 2 for testing. This dataset contains only 7 semantic
classes, as not all of
[Cityscapes](https://www.cityscapes-dataset.com/dataset-overview/#class-definitions)'
19 semantic classes are present.

Label Name     | Label ID
-------------- | --------
sidewalk       | 0
building       | 1
vegetation     | 2
sky            | 3
person&dagger; | 4
rider          | 5
bicycle        | 6
void           | 255

&dagger;: Single instance annotations are available.

### Prepare MOTChallenge-STEP for Training and Evaluation

#### Download data

1.  Download MOTChallenge images from https://motchallenge.net/
2.  Download groundtruth MOTChallenge-STEP panoptic maps from
    http://storage.googleapis.com/gresearch/tf-deeplab/data/motchallenge-step.tar.gz

The groundtruth panoptic map is encoded in the same way as described in
[KITTI-STEP dataset](./kitti_step.md).

#### Create tfrecord files

You should follow the same folder structure and run the command line script in
[KITTI-STEP dataset](./kitti_step.md) to prepare MOTChallenge-STEP dataset.

## Citing MOTChallenge-STEP

If you find this dataset helpful in your research, please use the following
BibTeX entry.

```
@article{step_2021,
  author={Mark Weber and Jun Xie and Maxwell Collins and Yukun Zhu and Paul Voigtlaender and Hartwig Adam and Bradley Green and Andreas Geiger and Bastian Leibe and Daniel Cremers and Aljosa Osep and Laura Leal-Taixe and Liang-Chieh Chen},
  title={{STEP}: Segmenting and Tracking Every Pixel},
  journal={arXiv:2102.11859},
  year={2021}
}
```
