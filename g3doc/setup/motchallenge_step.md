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

In the following, we provide a step-by-step walk through to prepare the data.

1.  Create the MOTChallenge-STEP directory:

    ```bash
    mkdir ${MOTCHALLENGE_STEP_ROOT}/images
    cd ${MOTCHALLENGE_STEP_ROOT}/images
    ```

2.  Download MOTChallenge images from https://motchallenge.net/data/MOTS.zip and
    unzip.

    ```bash
    wget ${MOTCHALLENGE_LINK}
    unzip ${MOTCHALLENGE_IMAGES}.zip
    ```

3.  Move and rename the data:

    ```bash
    # Create directories.
    mkdir train
    mkdir train/0002
    mkdir train/0009
    mkdir test
    mkdir test/0001
    mkdir test/0007

    # Copy data.
    cp -r MOTS/train/MOTS20-02/img1/* train/0002/
    cp -r MOTS/train/MOTS20-09/img1/* train/0009/
    cp -r MOTS/test/MOTS20-01/img1/* test/0001/
    cp -r MOTS/test/MOTS20-07/img1/* test/0007/

    # Clean up.
    rm -r MOTS
    ```

4.  Download groundtruth MOTChallenge-STEP panoptic maps from
    https://motchallenge.net/data/motchallenge-step.tar.gz

    ```bash
    cd ${MOTCHALLENGE_STEP_ROOT}
    wget ${MOTCHALLENGE_GT_LINK}
    tar -xvf ${MOTCHALLENGE_GT}.zip
    ```

The groundtruth panoptic map is encoded in the same way as described in
[KITTI-STEP dataset](./kitti_step.md).

DeepLab2 requires the dataset to be converted to TFRecords for efficient reading
and prefetching. To create the dataset for training and evaluation, run the
following command:

```bash
python deeplab2/data/build_step_data.py \
  --step_root=${MOTCHALLENGE_STEP_ROOT} \
  --output_dir=${OUTPUT_DIR}
```

This script outputs three sharded tfrecord files: `{train|test}@10.tfrecord`. In
the tfrecords, for `train` set, it contains the RGB image pixels as well as
their panoptic maps. For `test` set, it contains RGB images only. These files
will be used as the input for the model training and evaluation.

Optionally, you can also specify with `--use_two_frames` to encode two
consecutive frames into the tfrecord files.

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
