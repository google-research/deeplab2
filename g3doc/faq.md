# FAQ

________________________________________________________________________________

**Q1: What should I do if I encounter OOM (out-of-memory) while training the
models?**

**A1**: To avoid OOM, you could try:

1.  reducing the training crop size (i.e., the flag `crop_size` in
    `train_dataset_options`, and see Q2 for more details), which reduces the
    input size during training,

2.  using a larger output stride (e.g., 32) in the backbone (i.e., the flag
    `output_stride` in `model_options`, and see Q3 for more details), which
    reduces the usage of atrous convolution,

3.  using a smaller backbone, such as ResNet-50.

________________________________________________________________________________

**Q2: What is the `crop_size` I need to set?**

**A2**: DeepLab framework always uses `crop_size` equal to `output_stride` * k +
1, where k is an integer.

*   During inference/evaluation, since DeepLab framework uses whole-image
    inference, we need to set k so that the resulting `crop_size` (in
    `eval_dataset_options`) is slightly larger the largest image dimension in
    the dataset. For example, we set eval_crop_size = 1025x2049 for Cityscapes
    images whose image dimension is all equal to 1024x2048.

*   During training, we could set k to be any integer as long as it fits to your
    device memory. However, we notice a better performance when we have the same
    `crop_size` during training and evaluation (i.e., also use whole-image crop
    size during training).

________________________________________________________________________________

**Q3: What output stride should I use in the backbone?**

**A3**: Using a different output stride leads to a different accuracy-and-memory
trade-off. For example, DeepLabv1 uses output stride = 8, but it requires a lot
of device memory. In DeepLabv3+ paper, we found that using output stride = 16
strikes the best accuracy-and-memory trade-off, which is then our default
setting. If you wish to further reduce the memory usage, you could set output
stride to 32. Additionally, we suggest adjusting the `atrous_rates` in the ASPP
module as follows.

*   If `backbone.output_stride` = 32, use `atrous_rates` = [3, 6, 9].

*   If `backbone.output_stride` = 16, use `atrous_rates` = [6, 12, 18].

*   If `backbone.output_stride` = 8, use `atrous_rates` = [12, 24, 36].

Note that these settings may not be optimal. You may need to adjust them to
better fit your dataset.

________________________________________________________________________________

**Q4: Why are the results reported by the provided evaluation code slightly
different from the official evaluation code (e.g.,
[Cityscapes](https://github.com/mcordts/cityscapesScripts))?**

**A4**: In order to run everything end-to-end in the TensorFlow system (e.g.,
the on-line evaluation during training), we re-implemented the evaluation codes
in TensorFlow. Additionally, our whole system, including the training and
evaluation pipelines, uses the panoptic label format (i.e., `panoptic_label =
semantic_label * label_divisor + instance_id`, where the `label_divisor` should
be larger than the maximum number of instances per image), instead of the JSON
[COCO formats](https://cocodataset.org/#format-data). These two changes along
with rounding and similar issues result in some minor differences. Therefore,
our re-implemented evaluation code is mainly used for TensorFlow integration
(e.g., the support of on-line evaluation in TensorBoard). The users should run
the corresponding official evaluation code in order to compare with other
published papers. Note that all the reported numbers in our papers are evaluated
with the official evaluation code.

To facilitate the conversion between prediction formats, we also provide
instructions for running the official evaluation codes on
[Cityscapes](setup/cityscapes_test_server_evaluation.md) and
[COCO](setup/coco_test_server_evaluation.md).

________________________________________________________________________________

**Q5: What should I do, if I could not manage to compile TensorFlow along with
the provided efficient merging operation `merge_semantic_and_instance_maps`?**

**A5**: In this case, we provide another fallback solution, which implements the
merging operation with pure tf functions. This fallback solution does not
require any TensorFlow compilation. However, note that compared to our provided
TensorFlow merging operation `merge_semantic_and_instance_maps`, its inference
speed is slower and the resulting segmentation performance may also be slightly
lower.

To use the pure-tf-function version of `merge_semantic_and_instance_maps`, set
`merge_semantic_instance_with_tf_op` to `false` in your config's
`evaluator_options`.

________________________________________________________________________________

**Q6: What should I do, if I only want to train a semantic segmentation model?**

**A6**: While most of the tutorials focus on panoptic segmentation (unifying
semantic and instance segmentation), we understand that there are cases where
you would like to only train a semantic segmentation model. Currently, there are
two options to do that:

1.  Use DeepLabv3 or DeepLabv3+ model variants that could only perform semantic
    segmentation.

2.  Use Panoptic-DeepLab with only the semantic segmentation branch by disabling
    the instance segmentation branch. We provide one example config
    [here](../configs/cityscapes/panoptic_deeplab/resnet50_os32_semseg.textproto).
