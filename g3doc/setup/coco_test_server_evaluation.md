# Test Server Evaluation on COCO dataset

This page walks through the steps required to convert DeepLab2 predictions for
test server evaluation on [COCO](https://cocodataset.org/).

A high-level overview of the whole process:

1.  Save raw panoptic prediction in the two-channel format.

2.  Convert predictions in the two-channel format to the panoptic COCO format.

3.  Run local validation set evaluation or prepare test set evaluation.

We also define some environmental variables for simplicity and convenience:

`BASE_MODEL_DIRECTORY`: variables set in textproto file, which defines where all
checkpoints and results are saved.

`DATA_ROOT`: where the original COCO dataset is located.

`PATH_TO_SAVE`: where the converted results should be saved.

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

Then run the model in evaluation modes (with `--mode=eval`), and the results
will be saved at ${BASE_MODEL_DIRECTORY}/vis/raw_panoptic/\*.png.

## Convert the Prediction Format

Convert prediction results saved in the
[two-channel panoptic format](https://arxiv.org/pdf/1801.00868.pdf) to the
panoptic COCO format.

```bash
python panopticapi/converters/2channels2panoptic_coco_format.py \
  --source_folder=${BASE_MODEL_DIRECTORY}/vis/raw_panoptic \
  --images_json_file=${DATA_ROOT}/annotations/IMG_JSON \
  --categories_json_file=panopticapi/panoptic_coco_categories.json \
  --segmentations_folder=${PATH_TO_SAVE}/panoptic_cocoformat \
  --predictions_json_file=${PATH_TO_SAVE}/panoptic_cocoformat.json
```

The `IMG_JSON` refers to `panoptic_val2017.json` for *val* set and
`image_info_test-dev2017.json` for *test-dev* set.

## Run Local Evaluation Scripts (for *validation* set)

Run the [official scripts](https://github.com/cocodataset/panopticapi) to
evaluate validation set results.

```bash
python panopticapi/evaluation.py \
    --pred_json_file=${PATH_TO_SAVE}/panoptic_cocoformat.json \
    --pred_folder=${PATH_TO_SAVE}/panoptic_cocoformat \
    --gt_json_file=${DATA_ROOT}/annotations/panoptic_val2017.json \
    --gt_folder=${DATA_ROOT}/annotations/panoptic_val2017
```

## Prepare Submission Files (for *test* set)

Run the following command to prepare a submission file for test server
evaluation.

```bash
zip -r coco_test_submission_panoptic.zip ${PATH_TO_SAVE}/panoptic_cocoformat ${PATH_TO_SAVE}/panoptic_cocoformat.json
```
