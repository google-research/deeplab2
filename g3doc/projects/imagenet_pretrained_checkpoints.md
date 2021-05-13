# Download the pretrained checkpoints

To facilitate the model training, we also provide some checkpoints that are
pretrained on ImageNet.

After downloading the desired pretrained checkpoint, remember to update
the `initial_checkpoint` path in the config files.

## Checkpoints

**Simple Training Strategy**: This training strategy yields a similar
performance to the original ResNet paper [2].

Backbone | Pretrained Dataset
-------- | :---------------:
ResNet-50 ([initial_checkpoint](http://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_imagenet1k.tar.gz)) | ImageNet-1K

**Strong Training Strategy**: This training strategy additionally
employs AutoAugment [3], label-smoothing [4], and drop-path [5],  yielding
a stronger performance on ImageNet than the original ResNet paper [2].

Backbone | Pretrained Dataset
-------- | :---------------:
ResNet-50 ([initial_checkpoint](https://storage.googleapis.com/gresearch/tf-deeplab/checkpoint/resnet50_imagenet1k_strong_training_strategy.tar.gz)) | ImageNet-1K


### References

1. Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh,
   Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla,
   Michael Bernstein, Alexander C. Berg, and Li Fei-Fei. "ImageNet Large
   Scale Visual Recognition Challenge". IJCV, 2015.

2. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual
   learning for image recognition. In CVPR, 2016.

3. Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and
   Quoc V Le. "Autoaugment: Learning augmentation policies from data".
   In CVPR, 2019.

4. Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and
   Zbigniew Wojna. "Rethinking the inception architecture for computer
   vision." In CVPR, 2016.

5. Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger.
   "Deep networks with stochastic depth." In ECCV, 2016.
