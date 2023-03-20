# coding=utf-8
# Copyright 2023 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains Axial-ResNet model instances for Axial-DeepLab and MaX-DeepLab.

Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
    MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
"""

import abc
import collections.abc
import copy

from absl import logging
import tensorflow as tf

from deeplab2.model.encoder import axial_resnet


def _get_default_config():
  """Gets the default config for Axial-ResNets."""
  # The default config dictionary for an Axial-ResNet is the MaX-DeepLab-S
  # architecture for panoptic segmentation. This default config dictionary also
  # exactly matches the default arguments of the functions.
  default_config = {
      'num_blocks': [3, 4, 6, 3],
      'backbone_layer_multiplier': 1.0,
      'width_multiplier': 1.0,
      'stem_width_multiplier': 1.0,
      'output_stride': 16,
      'classification_mode': False,
      'backbone_type': 'resnet_beta',
      'use_axial_beyond_stride': 16,
      'backbone_use_transformer_beyond_stride': 32,
      'extra_decoder_use_transformer_beyond_stride': 32,
      'backbone_decoder_num_stacks': 0,
      'backbone_decoder_blocks_per_stage': 1,
      'extra_decoder_num_stacks': 0,
      'extra_decoder_blocks_per_stage': 1,
      'max_num_mask_slots': 128,
      'num_mask_slots': 128,
      'memory_channels': 256,
      'base_transformer_expansion': 1.0,
      'global_feed_forward_network_channels': 256,
      'high_resolution_output_stride': 4,
      'activation': 'relu',
      'block_group_config': {
          'attention_bottleneck_expansion': 2,
          'drop_path_keep_prob': 0.8,
          'drop_path_beyond_stride': 16,
          'drop_path_schedule': 'constant',
          'positional_encoding_type': None,
          'use_global_beyond_stride': 0,
          'use_sac_beyond_stride': 0,
          'use_squeeze_and_excite': False,
          'conv_use_recompute_grad': False,
          'axial_use_recompute_grad': True,
          'recompute_within_stride': 0,
          'transformer_use_recompute_grad': False,
          'axial_layer_config': {
              'query_shape': (129, 129),
              'key_expansion': 1,
              'value_expansion': 2,
              'memory_flange': (32, 32),
              'double_global_attention': False,
              'num_heads': 8,
              'use_query_rpe_similarity': True,
              'use_key_rpe_similarity': True,
              'use_content_similarity': True,
              'retrieve_value_rpe': True,
              'retrieve_value_content': True,
              'initialization_std_for_query_key_rpe': 1.0,
              'initialization_std_for_value_rpe': 1.0,
              'self_attention_activation': 'softmax',
          },
          'dual_path_transformer_layer_config': {
              'num_heads': 8,
              'bottleneck_expansion': 2,
              'key_expansion': 1,
              'value_expansion': 2,
              'feed_forward_network_channels': 2048,
              'use_memory_self_attention': True,
              'use_pixel2memory_feedback_attention': True,
              'transformer_activation': 'softmax',
          },
      },
      'bn_layer': tf.keras.layers.BatchNormalization,
      'conv_kernel_weight_decay': 0.0,
  }
  return default_config


def override(config_dict, override_dict):
  """Recursively overrides a config dict with another."""
  output_dict = copy.deepcopy(config_dict)
  for key, value in override_dict.items():  # pytype: disable=attribute-error  # class-cleanup
    if isinstance(value, collections.abc.Mapping):
      output_dict[key] = override(config_dict.get(key, {}), value)
    else:
      output_dict[key] = value
  return output_dict


class AxialResNetInstance(axial_resnet.AxialResNet):  # pytype: disable=ignored-abstractmethod  # abcmeta-check
  """A base Axial-ResNet model."""

  @classmethod
  @abc.abstractmethod
  def _get_config(cls):
    pass

  def __init__(self, name, **kwargs):
    """Builds an Axial-ResNet model."""
    # Get the config of the current model.
    current_config = self._get_config()

    # Override the default config with the current config. This line can be
    # omitted because the default config equals the default arguments of the
    # functions that build the model. But we make all the configs explicit here.
    current_config = override(_get_default_config(), current_config)

    # Finally, override the current model config with keyword arguments. In this
    # way, we still respect arguments passed as keyword arguments, such as
    # classification_mode, output_stride, etc.
    current_config = override(current_config, kwargs)
    logging.info('Axial-ResNet final config: %s', current_config)
    super(AxialResNetInstance, self).__init__(name, **current_config)


class MaXDeepLabS(AxialResNetInstance):
  """MaX-DeepLab-S for panoptic segmentation.

  Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
    MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    # Return an empty dictionary as the default values are all set for
    # MaX-DeepLab-S.
    return {}


class MaXDeepLabL(AxialResNetInstance):
  """MaX-DeepLab-L for panoptic segmentation.

  Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
    MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    return {
        'num_blocks': [3, 6, 3, 3],
        'backbone_type': 'wider_resnet',
        'backbone_use_transformer_beyond_stride': 16,
        'extra_decoder_use_transformer_beyond_stride': 16,
        'backbone_decoder_num_stacks': 1,
        'extra_decoder_num_stacks': 1,
        'extra_decoder_blocks_per_stage': 3,
        'memory_channels': 512,
        'base_transformer_expansion': 2.0,
        'global_feed_forward_network_channels': 512,
        'block_group_config': {
            'attention_bottleneck_expansion': 4,
            'drop_path_beyond_stride': 4,
            'axial_layer_config': {
                'key_expansion': 2,
                'value_expansion': 4,
            },
        },
    }


class MaXDeepLabSBackbone(MaXDeepLabS):
  """MaX-DeepLab-S backbone for image classification pretraining.

  Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
    MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    base_config = super(MaXDeepLabSBackbone, cls)._get_config()
    # Override the config of MaXDeepLabS.
    override_config = {
        'classification_mode': True,
        # The transformer blocks are not ImageNet pretrained. They are randomly
        # initialized and trained from scratch for panoptic segmentation.
        'backbone_use_transformer_beyond_stride': 0,
    }
    return override(base_config, override_config)


class MaXDeepLabLBackbone(MaXDeepLabL):
  """MaX-DeepLab-L backbone for image classification pretraining.

  Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
    MaX-DeepLab: "End-to-End Panoptic Segmentation with Mask Transformers",
      CVPR 2021. https://arxiv.org/abs/2012.00759
        Huiyu Wang, Yukun Zhu, Hartwig Adam, Alan Yuille, Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    base_config = super(MaXDeepLabLBackbone, cls)._get_config()
    # Override the config of MaXDeepLabL.
    override_config = {
        'classification_mode': True,
        # The transformer blocks are not ImageNet pretrained. They are randomly
        # initialized and trained from scratch for panoptic segmentation.
        'backbone_use_transformer_beyond_stride': 0,
    }
    return override(base_config, override_config)


class ResNet50(AxialResNetInstance):
  """A ResNet-50 instance.

  Note that the implementation is different from the original ResNet-50 in:
    (1) We apply strided convolutions in the first 3x3 convolution of the first
        residual block of a stage.
    (2) We replace the strided max pooling layer in the stem by applying strided
        convolution in the immediate next residual block.
  """

  @classmethod
  def _get_config(cls):
    return {
        'classification_mode': True,
        'backbone_type': 'resnet',
        'use_axial_beyond_stride': 0,
        'backbone_use_transformer_beyond_stride': 0,
        'block_group_config': {
            'drop_path_keep_prob': 1.0,
        },
    }


class ResNet50Beta(ResNet50):
  """A ResNet-50 but with inception stem.

  Note that the implementation is different from the original ResNet-50 in:
    (1) We apply strided convolutions in the first 3x3 convolution of the first
        residual block of a stage.
    (2) We replace the strided max pooling layer in the stem by applying strided
        convolution in the immediate next residual block.
  """

  @classmethod
  def _get_config(cls):
    base_config = super(ResNet50Beta, cls)._get_config()
    # Override the config of ResNet50.
    override_config = {
        'backbone_type': 'resnet_beta',
    }
    return override(base_config, override_config)


class AxialResNetL(ResNet50):
  """Axial-ResNet-L for image classification only.

  Axial-ResNet-L is a ResNet50 with use_axial_beyond_stride = 2.

  Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    base_config = super(AxialResNetL, cls)._get_config()
    # Override the config of ResNet50.
    override_config = {
        'use_axial_beyond_stride': 2,
    }
    return override(base_config, override_config)


class AxialResNetS(ResNet50):
  """Axial-ResNet-S for image classification only.

  Axial-ResNet-S is a ResNet50 with use_axial_beyond_stride = 2 and
  width_multiplier = 0.5.

  Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    base_config = super(AxialResNetS, cls)._get_config()
    # Override the config of ResNet50.
    override_config = {
        'width_multiplier': 0.5,
        'use_axial_beyond_stride': 2,
    }
    return override(base_config, override_config)


class AxialDeepLabL(ResNet50Beta):
  """Axial-DeepLab-L for panoptic segmentation.

  Axial-DeepLab-L is a ResNet50Beta with use_axial_beyond_stride = 2.
  Axial-DeepLab-L is also equivalent to Axial-ResNet-L with an inception stem.

  Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    base_config = super(AxialDeepLabL, cls)._get_config()
    override_config = {
        'use_axial_beyond_stride': 2,
    }
    return override(base_config, override_config)


class AxialDeepLabS(ResNet50Beta):
  """Axial-DeepLab-S for panoptic segmentation.

  Axial-DeepLab-S is a ResNet50Beta with use_axial_beyond_stride = 2 and
  width_multiplier = 0.5.
  Axial-DeepLab-S is also equivalent to Axial-ResNet-S with an inception stem.

  Reference:
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    base_config = super(AxialDeepLabS, cls)._get_config()
    override_config = {
        'width_multiplier': 0.5,
        'use_axial_beyond_stride': 2,
    }
    return override(base_config, override_config)


class SWideRNet(AxialResNetInstance):
  """A SWideRNet instance.

  Note that the implementation is different from the original SWideRNet in:
    (1) We apply strided convolutions in the first residual block of a stage,
        instead of the last residual block.
    (2) We replace the strided max pooling layer in the stem by applying strided
        convolution in the immediate next residual block.
    (3) We (optionally) use squeeze and excitation in all five stages, instead
        of the last four stages only.

  Reference:
    Scaling Wide Residual Networks for Panoptic Segmentation,
      https://arxiv.org/abs/2011.11675
        Liang-Chieh Chen, Huiyu Wang, Siyuan Qiao.
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    return {
        'num_blocks': [3, 6, 3, 3],
        'classification_mode': True,
        'backbone_type': 'wider_resnet',
        'use_axial_beyond_stride': 0,
        'backbone_use_transformer_beyond_stride': 0,
        'block_group_config': {
            'drop_path_beyond_stride': 4,
            'conv_use_recompute_grad': True,
        },
    }


class AxialSWideRNet(SWideRNet):
  """SWideRNet with axial attention blocks in the last two stages.

  Note that the implementation is different from the original SWideRNet in:
    (1) We apply strided convolutions in the first residual block of a stage,
        instead of the last residual block.
    (2) We replace the strided max pooling layer in the stem by applying strided
        convolution in the immediate next residual block.
    (3) We (optionally) use squeeze and excitation in all five stages, instead
        of the last four stages only.

  Reference:
    Scaling Wide Residual Networks for Panoptic Segmentation,
      https://arxiv.org/abs/2011.11675
        Liang-Chieh Chen, Huiyu Wang, Siyuan Qiao.
    Axial-Deeplab: Stand-Alone Axial-Attention for Panoptic Segmentation,
      ECCV 2020 Spotlight. https://arxiv.org/abs/2003.07853
        Huiyu Wang, Yukun Zhu, Bradley Green, Hartwig Adam, Alan Yuille,
          Liang-Chieh Chen.
  """

  @classmethod
  def _get_config(cls):
    base_config = super(AxialSWideRNet, cls)._get_config()
    override_config = {
        'use_axial_beyond_stride': 16,
        'block_group_config': {
            'attention_bottleneck_expansion': 4,
            'axial_layer_config': {
                'key_expansion': 2,
                'value_expansion': 4,
            },
        },
    }
    return override(base_config, override_config)


def get_model(name, **kwargs):
  """Gets the model instance given the model name."""
  name_lower = name.lower()
  if name_lower == 'max_deeplab_s':
    return MaXDeepLabS(name_lower, **kwargs)
  elif name_lower == 'max_deeplab_l':
    return MaXDeepLabL(name_lower, **kwargs)
  elif name_lower == 'max_deeplab_s_backbone':
    return MaXDeepLabSBackbone(name_lower, **kwargs)
  elif name_lower == 'max_deeplab_l_backbone':
    return MaXDeepLabLBackbone(name_lower, **kwargs)
  elif name_lower == 'resnet50':
    return ResNet50(name_lower, **kwargs)
  elif name_lower == 'resnet50_beta':
    return ResNet50Beta(name_lower, **kwargs)
  elif name_lower == 'swidernet' or name_lower == 'wide_resnet41':
    return SWideRNet(name_lower, **kwargs)
  elif name_lower == 'axial_swidernet':
    return AxialSWideRNet(name_lower, **kwargs)
  elif name_lower == 'axial_resnet_s':
    return AxialResNetS(name_lower, **kwargs)
  elif name_lower == 'axial_resnet_l':
    return AxialResNetL(name_lower, **kwargs)
  elif name_lower == 'axial_deeplab_s':
    return AxialDeepLabS(name_lower, **kwargs)
  elif name_lower == 'axial_deeplab_l':
    return AxialDeepLabL(name_lower, **kwargs)
  else:
    raise ValueError(name_lower + ' is not supported.')
