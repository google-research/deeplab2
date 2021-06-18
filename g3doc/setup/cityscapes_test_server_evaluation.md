# Test Server Evaluation on Cityscapes dataset

This page walks through the steps required to convert DeepLab2 predictions for
test server evaluation on [Cityscapes](https://www.cityscapes-dataset.com/).

A high-level overview of the whole process:

1.  Save raw panoptic prediction in the two-channel format.

2.  Create images json file.

3.  Convert predictions in the two-channel format to the panoptic COCO format.

4.  Run local validation set evaluation or prepare test set evaluation.

We also define some environmental variables for simplicity and convenience:

`EXPERIMENT_ROOT` and `EXPERIMENT_NAME`: variables set in textproto file, which
define where all checkpoints and results are saved.

`DATA_ROOT`: where the original Cityscapes dataset is located.

`PATH_TO_SAVE`: where the converted results should be saved.

`IMAGES_SPLIT`: *val* or *test* depending on the target split.

## Save Raw Panoptic Prediction

Save the raw panoptic predictions in the
[two-channel panoptic format](https://arxiv.org/pdf/1801.00868.pdf) by ensuring
the following fields are set properly in the textproto config file.

```
eval_dataset_options.decode_groundtruth_label = false
evaluator_options.save_predictions = true
evaluator_options.save_raw_predictions = true
evaluator_options.convert_raw_to_eval_ids = true
```

Then run the model in evaluation modes (with `--job_type="eval"`), and the
results will be saved at
${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}/vis/raw_panoptic/\*.png.

## Create Images JSON

Create images json file by running the following commands.

```bash
python deeplab2/utils/create_images_json_for_cityscapes.py \
  --image_dir=${DATA_ROOT}/leftImg8bit/${IMAGES_SPLIT} \
  --output_json_path=${PATH_TO_SAVE}/${IMAGES_SPLIT}_images.json \
  --only_basename \
  --include_image_type_suffix=false
```

## Convert the Prediction Format

Convert prediction results saved in the
[two-channel panoptic format](https://arxiv.org/pdf/1801.00868.pdf) to the
panoptic COCO format.

```bash
python panopticapi/converters/2channels2panoptic_coco_format.py \
  --source_folder=${EXPERIMENT_ROOT}/${EXPERIMENT_NAME}/vis/raw_panoptic \
  --images_json_file=${PATH_TO_SAVE}/${IMAGES_SPLIT}_images.json\
  --categories_json_file=deeplab2/utils/panoptic_cityscapes_categories.json \
  --segmentations_folder=${PATH_TO_SAVE}/panoptic_cocoformat \
  --predictions_json_file=${PATH_TO_SAVE}/panoptic_cocoformat.json
```

## Run Local Evaluation Scripts (for *validation* set)

Run the [official scripts](https://github.com/mcordts/cityscapesScripts) to
evaluate validation set results.

```bash
python cityscapesscripts/evaluation/evalPanopticSemanticLabeling.py \
    --prediction-json_file=${PATH_TO_SAVE}/panoptic_cocoformat.json \
    --prediction-folder=${PATH_TO_SAVE}/panoptic_cocoformat \
    --gt-json-file=${DATA_ROOT}/gtFine/cityscapes_panoptic_val.json \
    --gt-folder=${DATA_ROOT}/gtFine/cityscapes_panoptic_val
```

## Prepare Submission Files (for *test* set)

Run the following command to prepare a submission file for test server
evaluation.

```bash
zip -r cityscapes_test_submission_panoptic.zip ${PATH_TO_SAVE}/panoptic_cocoformat ${PATH_TO_SAVE}/panoptic_cocoformat.json
```
