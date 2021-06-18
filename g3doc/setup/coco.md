# Run DeepLab2 on COCO dataset

This page walks through the steps required to generate
[COCO](https://cocodataset.org/) panoptic segmentation data for DeepLab2.
DeepLab2 uses sharded TFRecords for efficient processing of the data.

## Prework

Before running any Deeplab2 scripts, the users should (1) access the
[COCO dataset website](https://cocodataset.org/) to download the dataset,
including [2017 Train images](http://images.cocodataset.org/zips/train2017.zip),
[2017 Val images](http://images.cocodataset.org/zips/val2017.zip),
[2017 Test images](http://images.cocodataset.org/zips/test2017.zip), and
[2017 Panoptic Train/Val annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip),
and (2) unzip the downloaded files.

After finishing above steps, the expected directory structure should be as
follows:

```
.(COCO_ROOT)
+-- train2017
|   |
|   +-- *.jpg
|
|-- val2017
|   |
|   +-- *.jpg
|
|-- test2017
|   |
|   +-- *.jpg
|
+-- annotations
     |
     +-- panoptic_{train|val}2017.json
     +-- panoptic_{train|val}2017
```

## Convert prepared dataset to TFRecord

Use the following commandline to generate COCO TFRecords:

```bash
# For generating data for panoptic segmentation task
python deeplab2/data/build_coco_data.py \
  --coco_root=${COCO_ROOT} \
  --output_dir=${OUTPUT_DIR}
```

Commandline above will output three sharded tfrecord files:
`{train|val|test}@1000.tfrecord`. In the tfrecords, for `train` and `val` set,
it contains the RGB image pixels as well as corresponding annotations. For
`test` set, it contains RGB images only. These files will be used as the input
for the model training and evaluation.

Note that we map the class ID to continuous IDs. Specifically, we map the
original label ID, which ranges from 1 to 200, to the contiguous ones ranging
from 1 to 133.

### TFExample proto format for COCO

The Example proto contains the following fields:

*   `image/encoded`: encoded image content.
*   `image/filename`: image filename.
*   `image/format`: image file format.
*   `image/height`: image height.
*   `image/width`: image width.
*   `image/channels`: image channels.
*   `image/segmentation/class/encoded`: encoded segmentation content.
*   `image/segmentation/class/format`: segmentation encoding format.

For panoptic segmentation, the encoded segmentation map will be the raw bytes of
an int32 panoptic map, where each pixel is assigned to a panoptic ID, which is
computed by:

```
  panoptic ID = semantic ID * label divisor + instance ID
```

where semantic ID will be:

*   ignore label (0) for pixels not belonging to any segment
*   for segments associated with `iscrowd` label:
    *   (default): ignore label (0)
    *   (if set `--treat_crowd_as_ignore=false` while running
        `build_coco_data.py`): `category_id`
*   `category_id` for other segments

The instance ID will be 0 for pixels belonging to

*   `stuff` class
*   `thing` class with `iscrowd` label
*   pixels with ignore label

and `[1, label divisor)` otherwise.
