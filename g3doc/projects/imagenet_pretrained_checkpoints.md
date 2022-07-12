# Download the pretrained checkpoints

To facilitate the model training, we also provide some checkpoints that are
pretrained on ImageNet.

After downloading the desired pretrained checkpoint, remember to update
the `initial_checkpoint` path in the config files.

## Checkpoints

**Simple training strategy**: This training strategy yields a similar
performance to the original ResNet paper [2], and MobileNetV3 paper [6].

Backbone                                                                                                                                      | Pretrained Dataset
--------------------------------------------------------------------------------------------------------------------------------------------- | :----------------:
ResNet-50 ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_imagenet1k.tar.gz))                   | ImageNet-1K
MobileNetV3-Small ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/mobilenet_v3_small_imagenet1k.tar.gz)) | ImageNet-1K
MobileNetV3-Large ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/mobilenet_v3_large_imagenet1k.tar.gz)) | ImageNet-1K

**Strong training strategy**: This training strategy additionally
employs AutoAugment [3], label-smoothing [4], and drop-path [5],  yielding
a stronger performance on ImageNet than the original ResNet paper [2].

Backbone                                                                                                                                                                                            | Pretrained Dataset
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :----------------:
ResNet-50 ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_imagenet1k_strong_training_strategy.tar.gz))                                                | ImageNet-1K
ResNet-50-Beta ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_beta_imagenet1k_strong_training_strategy.tar.gz))                                      | ImageNet-1K
Wide-ResNet-41 ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/wide_resnet41_imagenet1k_strong_training_strategy.tar.gz))                                      | ImageNet-1K
SWideRNet-SAC-(1, 1, 1) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/swidernet_sac_1_1_1_imagenet1k_strong_training_strategy.tar.gz))                       | ImageNet-1K
SWideRNet-SAC-(1, 1, 3) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/swidernet_sac_1_1_3_imagenet1k_strong_training_strategy.tar.gz))                       | ImageNet-1K
SWideRNet-SAC-(1, 1, 4.5) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/swidernet_sac_1_1_4.5_imagenet1k_strong_training_strategy.tar.gz))                   | ImageNet-1K
Axial-SWideRNet-(1, 1, 1) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/axial_swidernet_1_1_1_imagenet1k_strong_training_strategy.tar.gz))                   | ImageNet-1K
Axial-SWideRNet-(1, 1, 3) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/axial_swidernet_1_1_3_imagenet1k_strong_training_strategy.tar.gz))                   | ImageNet-1K
Axial-SWideRNet-(1, 1, 4.5) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/axial_swidernet_1_1_4.5_imagenet1k_strong_training_strategy.tar.gz))               | ImageNet-1K
MaX-DeepLab-S-Backbone (aka MaX-S w/ ReLU) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/max_deeplab_s_backbone_imagenet1k_strong_training_strategy.tar.gz)) | ImageNet-1K
MaX-DeepLab-L-Backbone (aka MaX-L w/ ReLU) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/max_deeplab_l_backbone_imagenet1k_strong_training_strategy.tar.gz)) | ImageNet-1K
Axial-ResNet-50 (aka MaX-S w/ GeLU) ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/maxs_gelu_imagenet_22k_1k_224.tar.gz))                                     | ImageNet-22K + 1K

**Converted from official weights**: This section includes pretrained weights
for ConvNeXt [7], which are converted from the official PyTorch version [8].

Backbone                                                                                                                                    | Pretrained Dataset
------------------------------------------------------------------------------------------------------------------------------------------- | :----------------:
ConvNeXt-T ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/convnext_tiny_imagenet_22k_1k_384.tar.gz))  | ImageNet-22K + 1K
ConvNeXt-S ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/convnext_small_imagenet_22k_1k_384.tar.gz)) | ImageNet-22K + 1K
ConvNeXt-B ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/convnext_base_imagenet_22k_1k_384.tar.gz))  | ImageNet-22K + 1K
ConvNeXt-L ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/convnext_large_imagenet_22k_1k_384.tar.gz)) | ImageNet-22K + 1K

### References

1.  Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean
    Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein,
    Alexander C. Berg, and Li Fei-Fei. "ImageNet Large Scale Visual Recognition
    Challenge". IJCV, 2015.

2.  Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual
    learning for image recognition. In CVPR, 2016.

3.  Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le.
    "Autoaugment: Learning augmentation policies from data". In CVPR, 2019.

4.  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and
    Zbigniew Wojna. "Rethinking the inception architecture for computer vision."
    In CVPR, 2016.

5.  Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. "Deep
    networks with stochastic depth." In ECCV, 2016.

6.  Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing
    Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and
    Hartwig Adam. "Searching for mobilenetv3." In ICCV, 2019.

7.  Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor
    Darrell, and Saining Xie. "A ConvNet for the 2020s." In CVPR, 2022.

8.  https://github.com/facebookresearch/ConvNeXt
