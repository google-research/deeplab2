# Run DeepLab2 on ADE20K dataset

This page walks through the steps required to generate
[ADE20K](https://groups.csail.mit.edu/vision/datasets/ADE20K/) panoptic
segmentation data for DeepLab2.

## Prework

Before running any Deeplab2 scripts, the users should (1) access the
[ADE20K dataset website](https://groups.csail.mit.edu/vision/datasets/ADE20K/)
to download the dataset, and (2) prepare the panoptic annotation using
[Mask2Former's script](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/prepare_ade20k_pan_seg.py).

After finishing above steps, the expected directory structure should be as
follows:

```
.(ADE20K_ROOT)
+-- images
|
|-- annotations
|
|-- objectInfo150.txt
|
|-- annotations_instance
|
|-- ade20k_panoptic_{train,val}.json
|
+-- ade20k_panoptic_{train,val}
```

## Convert prepared dataset to TFRecord

Use the following commandline to generate ADE20K TFRecords:

```bash
# For generating data for panoptic segmentation task
python deeplab2/data/build_ade20k_data.py \
  --ade20k_root=${ADE20K_ROOT} \
  --output_dir=${OUTPUT_DIR}
```

Commandline above will output two sharded tfrecord files:
`{train|val}@1000.tfrecord`. In the tfrecords, for `train` and `val` set, it
contains the RGB image pixels as well as corresponding annotations. These files
will be used as the input for the model training and evaluation.

### TFExample proto format for ADE20K

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
*   `category_id` for other segments

The instance ID will be 0 for pixels belonging to

*   `stuff` class
*   `thing` class with `iscrowd` label
*   pixels with ignore label

and `[1, label divisor)` otherwise.
