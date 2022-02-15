# Convert your own dataset for DeepLab2 framework

You may want to train DeepLab2 on your own dataset. Here, we provide some
guidances and hopefully that will facillitate the preparation process.

1.  Prepare your own dataset.
    *   **Images** should be stored either in `jpg` or `png` format.
    *   **Annotations** should be stored either in `png` or `json` format. The
        DeepLab2 framework assumes the panoptic label format (i.e.,
        `panoptic_label = semantic_label * label_divisor + instance_id`, where
        the `label_divisor` should be larger than the maximum number of
        instances per image).
        *   The `png` format refers to the case where we could split semantic
            label and instance id to RGB channels. For example, R-channel stores
            semantic label, while G- and B-channel store instance id (G:
            instance_id // 256 and B: instance_id % 256).
        *   The `json` format refers to the
            [COCO panoptic json format](https://cocodataset.org/#format-data).
2.  Convert the dataset to TFRecord.

    *   Update our provided example code (e.g.,
        [build_step_data.py](../../data/build_step_data.py) for `png` format,
        and [build_coco_data.py](../../data/build_coco_data.py) for `json`
        format) to convert your dataset to TFRecord.
    *   Alternatively, if you are using your own binary to create TFRecords,
        make sure to include the same fields in the proto as what our example
        code creates.

3.  Modify the `dataset.py` (path: `${DEEPLAB2}/data/dataset.py`) to provide
    your dataset information.

    *   Set the `panoptic_label_divisor` (i.e., the `label_divisor` above)
        correctly. Its value should be larger than the maximum number of
        instances that could appear per image in your dataset.
    *   Set the `ignore_label` properly. Pixels annotated with `ignore_label`
        are not used during both training and evaluation. If your dataset does
        not contain the `ignore_label` annotations, you could simply set it to
        be a large value (e.g., 255 as for
        [Cityscapes](https://www.cityscapes-dataset.com/)).
    *   Set the `class_has_instance_list` properly. The variable specifies which
        class belongs to the `thing` class (i.e., countable objects such as
        people, cars).
    *   Set the colormap (for visualization) properly. You may also need to
        define your own colormap (see `${DEEPLAB2}/trainer/vis_utils.py`).

4.  Prepare the experiment config.

    *   Update our provided example configs (path:
        `${DEEPLAB2}/configs/${DATASET}/${MODEL}/${BACKBONE}`) for your use
        case. A few things that may worth your attention:
        *   Set the `crop_size` correctly for both training and evaluation. See
            Q2 in [FAQ](../faq.md) for more details.
        *   Tune the config flags for your dataset (e.g., `base_learning_rate`,
            `training_number_of_step`, and so on).

Finally, if your dataset only contains semantic segmentation annotations,
you could still use DeepLab2 framework with some minor changes:

1. Since the code only reads panoptic data at the moment, you need to set
`panoptic_label_divisor = k`, where k is any positive integer,
`instance_id = 0`, and `class_has_instances_list = []` (i.e., we treat the
dataset as the one that contains only `stuff` classes), when you are (1)
converting the dataset to TFRecord (e.g.,
[build_step_data.py](../../data/build_step_data.py)),
and (2) adding dataset information in dataset.py.
2. Have a config similar to
`${DEEPLAB2}/configs/cityscapes/panoptic_deeplab/resnet50_os32_semseg.textproto`
, where the instance branch is not
initiated.

At this point, you are good to go! Enjoy training DeepLab2!
