# Using DeepLab2

In the following, we provide instructions on how to run DeepLab2.

## Prerequisites

We assume DeepLab2 is successfully installed and the necessary datasets are
configured.

*   See [Installation](installation.md).
*   See dataset guides:
    *   [Cityscapes](cityscapes.md).
    *   [KITTI-STEP](kitti_step.md).
    *   [and many more](./).

## Running DeepLab2

DeepLab2 contains several implementations of state-of-the-art methods. In the
following, we discuss all steps from choosing a model, setting up the
configuration to training and evaluating it.

### Choosing a model

For this tutorial, we use Panoptic-DeepLab, however, running any other model
follows the same steps. For each network architecture, we provide a guide that
contains example configurations and (pretrained) checkpoints. You can find all
guides [here](../projects/). For now, please checkout
[Panoptic-DeepLab](../projects/panoptic_deeplab.md).

We will use the Resnet50 model as an example for this guide. If you just want to
run the network without training, please download the corresponding checkpoint
trained by us. If you would like to train the network, please download the
corresponding ImageNet pretrained checkpoint from
[here](../projects/imagenet_pretrained_checkpoints.md).

### Defining a configuration

When you want to train or evaluate a network, DeepLab2 requires a corresponding
configuration. This configuration contains information about the network
architecture as well as all sorts of hyper-parameters. Fortunately, for almost
all settings we provide default values and example configurations. The
configuration of Panoptic-DeepLab with ResNet50 for the Cityscapes dataset can
be found
[here](../../configs/cityscapes/panoptic_deeplab/resnet50_os32_merge_with_pure_tf_func.textproto).

Using our default parameters there are only a few things that needs to be
defined:

1.  The name of the experiment `experiment_name`. The experiment name is used as
    a folder name to store all experiment related files in.
2.  The initial checkpoint `initial_checkpoint`, which can be an empty string
    for none or the path to a checkpoint (e.g., pretrained on ImageNet or fully
    trained by us.)
3.  The training dataset `train_dataset_options.file_pattern`, which should
    point to the TFRecords of the Cityscapes train set.
4.  The evaluation dataset `eval_dataset_options.file_pattern`, which should
    point to the TFRecords of the Cityscapes val set.
5.  If the custom CUDA kernel is successfully compiled, we recommend to set
    `merge_semantic_and_instance_with_tf_op` to true.

For a detailed explanation of all the parameters, we refer to the documented
definitions of the proto files. A good starting place is the
[config.proto](../../config.proto). The `ExperimentOptions` are a collection of
all necessary configurations ranging from the model architecture to the training
settings.

### Training and Evaluating

We currently support four different modes to run DeepLab2:

*   Training: This will only train the network based on the provided
    configuration.
*   Evaluation: This will only evaluate the network based on the provided
    configuration.
*   Continuous Evaluation: This mode will constantly monitor a directory for
    newly saved checkpoints that will be evaluated until a timeout. This mode is
    useful when runing separate jobs for training and evaluation (e.g., a multi
    GPU job for training, and a single GPU job for evaluating).
*   Interleaved Training and Evaluation: In this mode, training and evaluation
    will run interleaved. This is not supported for multi GPU jobs.

### Putting everything together

To run DeepLab2 on GPUs, the following command should be used:

```bash
python trainer/train.py \
    --config_file=${CONFIG_FILE} \
    --mode={train | eval | train_and_eval | continuous_eval} \
    --model_dir=${BASE_MODEL_DIRECTORY} \
    --num_gpus=${NUM_GPUS}
```

You can also launch DeepLab2 on TPUS. For this, the TPU address needs to be
specified:

```bash
python trainer/train.py \
    --config_file=${CONFIG_FILE} \
    --mode={train | eval | train_and_eval | continuous_eval} \
    --model_dir=${BASE_MODEL_DIRECTORY} \
    --master=${TPU_ADDRESS}
```

For a detailed explanation of each option run:

```bash
python trainer/train.py --help
```
