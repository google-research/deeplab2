# Run DeepLab2 on Cityscapes dataset

This page walks through the steps required to generate
[Cityscapes](https://www.cityscapes-dataset.com/) data for DeepLab2. DeepLab2
uses sharded TFRecords for efficient processing of the data.

## Prework

Before running any Deeplab2 scripts, the user should 1. register on the
Cityscapes dataset [website](https://www.cityscapes-dataset.com) to download the
dataset (gtFine_trainvaltest.zip and leftImg8bit_trainvaltest.zip). 2. install
cityscapesscripts via pip: `bash # This will install the cityscapes scripts and
its stand-alone tools. pip install cityscapesscripts`

1.  run the tools provided by Cityscapes to generate the training groundtruth.
    See sample commandlines below:

```bash
  # Set CITYSCAPES_DATASET to your dataset root.

  # Create train ID label images.
  CITYSCAPES_DATASET='.' csCreateTrainIdLabelImgs

  # To generate panoptic groundtruth, run the following command.
  CITYSCAPES_DATASET='.' csCreatePanopticImgs --use-train-id

  # [Optional] Generate panoptic groundtruth with EvalId to match evaluation
  # on the server. This step is not required for generating TFRecords.
  CITYSCAPES_DATASET='.' csCreatePanopticImgs
```

After running above commandlines, the expected directory structure should be as
follows:

```
cityscapes
+-- gtFine
|   |
|   +-- train
|   |   |
|   |   +-- aachen
|   |       |
|   |       +-- *_color.png
|   |       +-- *_instanceIds.png
|   |       +-- *_labelIds.png
|   |       +-- *_polygons.json
|   |       +-- *_labelTrainIds.png
|   |   ...
|   +-- val
|   +-- test
|   +-- cityscapes_panoptic_{train|val|test}_trainId.json
|   +-- cityscapes_panoptic_{train|val|test}_trainId
|   |   |
|   |   +-- *_panoptic.png
|   +-- cityscapes_panoptic_{train|val|test}.json
|   +-- cityscapes_panoptic_{train|val|test}
|       |
|       +-- *_panoptic.png
|
+-- leftImg8bit
     |
     +-- train
     +-- val
     +-- test
```

## Convert prepared dataset to TFRecord

Note: the rest of this doc and released DeepLab2 models use `TrainId` instead of
`EvalId` (which is used on the evaluation server). For evaluation on the server,
you would need to convert the predicted labels to `EvalId` .

Use the following commandline to generate cityscapes TFRecords:

```bash
# Assuming we are under the folder where deeplab2 is cloned to:

# For generating data for semantic segmentation task only
python deeplab2/data/build_cityscapes_data.py \
  --cityscapes_root=${PATH_TO_CITYSCAPES_ROOT} \
  --output_dir=${OUTPUT_PATH_FOR_SEMANTIC} \
  --create_panoptic_data=false

# For generating data for panoptic segmentation task
python deeplab2/data/build_cityscapes_data.py \
  --cityscapes_root=${PATH_TO_CITYSCAPES_ROOT} \
  --output_dir=${OUTPUT_PATH_FOR_PANOPTIC}
```

Commandline above will output three sharded tfrecord files:
`{train|val|test}@10.tfrecord`. In the tfrecords, for `train` and `val` set, it
contains the RGB image pixels as well as corresponding annotations. For `test`
set, it contains RGB images only. These files will be used as the input for the
model training and evaluation.

### TFExample proto format for cityscapes

The Example proto contains the following fields:

*   `image/encoded`: encoded image content.
*   `image/filename`: image filename.
*   `image/format`: image file format.
*   `image/height`: image height.
*   `image/width`: image width.
*   `image/channels`: image channels.
*   `image/segmentation/class/encoded`: encoded segmentation content.
*   `image/segmentation/class/format`: segmentation encoding format.

For semantic segmentation (`--create_panoptic_data=false`), the encoded
segmentation map will be the same as PNG file created by
`createTrainIdLabelImgs.py`.

For panoptic segmentation, the encoded segmentation map will be the raw bytes of
a int32 panoptic map, where each pixel is assigned to a panoptic ID. Unlike the
ID used in Cityscapes script (`json2instanceImg.py`), this panoptic ID is
computed by:

```
  panoptic ID = semantic ID * label divisor + instance ID
```

where semantic ID will be:

*   ignore label (255) for pixels not belonging to any segment
*   for segments associated with `iscrowd` label:
    *   (default): ignore label (255)
    *   (if set `--treat_crowd_as_ignore=false` while running
        `build_cityscapes_data.py`): `category_id` (use TrainId)
*   `category_id` (use TrainId) for other segments

The instance ID will be 0 for pixels belonging to

*   `stuff` class
*   `thing` class with `iscrowd` label
*   pixels with ignore label

and `[1, label divisor)` otherwise.
