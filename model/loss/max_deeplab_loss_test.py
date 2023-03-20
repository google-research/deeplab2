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

"""Tests for max_deeplab_loss.py."""

import tensorflow as tf

from deeplab2 import common
from deeplab2 import trainer_pb2
from deeplab2.data import dataset
from deeplab2.model.loss import max_deeplab_loss


class MaXDeepLabLossTest(tf.test.TestCase):

  def test_max_deeplab_loss(self):
    # Build the loss layer.
    dataset_info = dataset.COCO_PANOPTIC_INFORMATION
    semantic_loss_options = trainer_pb2.LossOptions.SingleLossOptions(
        name='softmax_cross_entropy')
    pq_style_loss_options = trainer_pb2.LossOptions.SingleLossOptions()
    mask_id_cross_entropy_loss_options = (
        trainer_pb2.LossOptions.SingleLossOptions())
    instance_discrimination_loss_options = (
        trainer_pb2.LossOptions.SingleLossOptions())
    loss_options_1 = trainer_pb2.LossOptions(
        semantic_loss=semantic_loss_options,
        pq_style_loss=pq_style_loss_options,
        mask_id_cross_entropy_loss=mask_id_cross_entropy_loss_options,
        instance_discrimination_loss=instance_discrimination_loss_options)
    loss_layer_1 = max_deeplab_loss.MaXDeepLabLoss(
        loss_options_1,
        num_classes=dataset_info.num_classes,
        ignore_label=dataset_info.ignore_label,
        thing_class_ids=dataset_info.class_has_instances_list,
        auxiliary_output_number=0)
    loss_options_2 = trainer_pb2.LossOptions(
        pq_style_loss=pq_style_loss_options)
    loss_layer_2 = max_deeplab_loss.MaXDeepLabLoss(
        loss_options_2,
        num_classes=dataset_info.num_classes,
        ignore_label=dataset_info.ignore_label,
        thing_class_ids=dataset_info.class_has_instances_list,
        auxiliary_output_number=1)

    # Build the inputs.
    aux_pred_dict = {
        common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY:
            tf.random.uniform(shape=[2, 9, 9, 8]),
        common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY:
                tf.random.uniform(shape=[2, 9, 9, 128]),
        common.PRED_TRANSFORMER_CLASS_LOGITS_KEY:
                tf.random.uniform(shape=[2, 128, 134]),
    }
    pred_dict = {
        common.PRED_SEMANTIC_LOGITS_KEY:
            tf.random.uniform(shape=[2, 9, 9, 134]),
        common.PRED_PIXEL_SPACE_NORMALIZED_FEATURE_KEY:
            tf.random.uniform(shape=[2, 9, 9, 8]),
        common.PRED_PIXEL_SPACE_MASK_LOGITS_KEY:
                tf.random.uniform(shape=[2, 9, 9, 128]),
        common.PRED_TRANSFORMER_CLASS_LOGITS_KEY:
                tf.random.uniform(shape=[2, 128, 134]),
        common.PRED_AUXILIARY_OUTPUTS: (aux_pred_dict,),
    }
    gt_dict = {
        common.GT_SEMANTIC_KEY:
            tf.ones(shape=[2, 33, 33], dtype=tf.int32),
        common.SEMANTIC_LOSS_WEIGHT_KEY:
            tf.ones(shape=[2, 33, 33], dtype=tf.float32),
        common.GT_THING_ID_MASK_KEY:
            tf.ones(shape=[2, 33, 33], dtype=tf.int32),
        common.GT_THING_ID_CLASS_KEY:
            tf.concat(
                # An image with ten people (class_id = 1) and 118 void masks.
                [
                    tf.ones(shape=[2, 10], dtype=tf.int32),
                    -tf.ones(shape=[2, 118], dtype=tf.int32)
                ],
                axis=-1),
    }
    loss_dict_1 = loss_layer_1((gt_dict, pred_dict))

    self.assertIn(common.SEMANTIC_LOSS, loss_dict_1)
    self.assertIn(common.PQ_STYLE_LOSS_CLASS_TERM, loss_dict_1)
    self.assertIn(common.PQ_STYLE_LOSS_MASK_DICE_TERM, loss_dict_1)
    self.assertIn(common.MASK_ID_CROSS_ENTROPY_LOSS, loss_dict_1)
    self.assertIn(common.INSTANCE_DISCRIMINATION_LOSS, loss_dict_1)
    self.assertNotIn(common.PQ_STYLE_LOSS, loss_dict_1)

    self.assertIn(common.PQ_STYLE_LOSS_CLASS_TERM, loss_layer_1.loss_terms)
    self.assertIn(common.PQ_STYLE_LOSS_MASK_DICE_TERM, loss_layer_1.loss_terms)
    self.assertIn(common.MASK_ID_CROSS_ENTROPY_LOSS, loss_layer_1.loss_terms)
    self.assertIn(common.INSTANCE_DISCRIMINATION_LOSS, loss_layer_1.loss_terms)
    self.assertNotIn(common.PQ_STYLE_LOSS, loss_layer_1.loss_terms)

    loss_dict_2 = loss_layer_2((gt_dict, pred_dict))

    self.assertIn(common.PQ_STYLE_LOSS_CLASS_TERM, loss_dict_2)
    self.assertIn(common.PQ_STYLE_LOSS_MASK_DICE_TERM, loss_dict_2)
    self.assertNotIn(common.MASK_ID_CROSS_ENTROPY_LOSS, loss_dict_2)
    self.assertNotIn(common.INSTANCE_DISCRIMINATION_LOSS, loss_dict_2)
    self.assertNotIn(common.PQ_STYLE_LOSS, loss_dict_2)

    self.assertIn(common.PQ_STYLE_LOSS_CLASS_TERM, loss_layer_2.loss_terms)
    self.assertIn(common.PQ_STYLE_LOSS_MASK_DICE_TERM, loss_layer_2.loss_terms)
    self.assertNotIn(common.MASK_ID_CROSS_ENTROPY_LOSS, loss_layer_2.loss_terms)
    self.assertNotIn(common.INSTANCE_DISCRIMINATION_LOSS,
                     loss_layer_2.loss_terms)
    self.assertNotIn(common.PQ_STYLE_LOSS, loss_layer_2.loss_terms)

    self.assertIn('aux_0' + common.PQ_STYLE_LOSS_CLASS_TERM, loss_dict_2)
    self.assertIn('aux_0' + common.PQ_STYLE_LOSS_MASK_DICE_TERM, loss_dict_2)
    self.assertNotIn('aux_0' + common.MASK_ID_CROSS_ENTROPY_LOSS, loss_dict_2)
    self.assertNotIn('aux_0' + common.INSTANCE_DISCRIMINATION_LOSS, loss_dict_2)
    self.assertNotIn('aux_0' + common.PQ_STYLE_LOSS, loss_dict_2)

    self.assertIn('aux_0' + common.PQ_STYLE_LOSS_CLASS_TERM,
                  loss_layer_2.loss_terms)
    self.assertIn('aux_0' + common.PQ_STYLE_LOSS_MASK_DICE_TERM,
                  loss_layer_2.loss_terms)
    self.assertNotIn('aux_0' + common.MASK_ID_CROSS_ENTROPY_LOSS,
                     loss_layer_2.loss_terms)
    self.assertNotIn('aux_0' + common.INSTANCE_DISCRIMINATION_LOSS,
                     loss_layer_2.loss_terms)
    self.assertNotIn('aux_0' + common.PQ_STYLE_LOSS, loss_layer_2.loss_terms)


if __name__ == '__main__':
  tf.test.main()
