// Copyright 2022 The Deeplab2 Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include /*third_party*/"tensorflow/core/framework/op.h"
#include /*third_party*/"tensorflow/core/framework/shape_inference.h"

namespace tensorflow_models {
namespace deeplab {
namespace deeplab2 {

using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

REGISTER_OP("MergeSemanticAndInstanceMaps")
    .Input("semantic_maps: int32")
    .Input("instance_maps: int32")
    .Input("thing_ids: int32")
    .Attr("label_divisor: int = 256")
    .Attr("stuff_area_limit: int = 0")
    .Attr("void_label: int = 0")
    .Output("parsing_maps: int32")
    .SetShapeFn([](InferenceContext* c) {
      ShapeHandle semantic_maps;
      ShapeHandle instance_maps;
      ShapeHandle thing_ids;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &semantic_maps));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &instance_maps));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &thing_ids));
      DimensionHandle batch = c->Dim(semantic_maps, 0);
      DimensionHandle height = c->Dim(semantic_maps, 1);
      DimensionHandle width = c->Dim(semantic_maps, 2);
      c->set_output(0, c->MakeShape({batch, height, width}));
      return tensorflow::Status::OK();
    })
    .Doc(R"doc(
Generates parsing maps from semantic maps and instance maps.

Parsing maps, or panoptic segmentation, are merged from the predicted semantic
maps and class-agnostic instance maps. This function merges the maps in the
following way:

1) If a pixel belongs to `stuff` class (e.g., sky), the function directly uses
  the semantic label from the semantic map and uses 0 as the instance label.
2) If a pixel belongs to `thing` class (e.g., person), it uses the instance
  label from the instance map and uses the majority of the semantic labels of
  the same instance as the final semantic label.
3) The function relabels each instance, so that the instance label of each
  semantic class is in the range of [1, num_instances_of_the_semantic_class].

Note that this operation is first poposed in the DeeperLab paper and adopted
by the Panoptic-DeepLab framework.
  - DeeperLab: Single-Shot Image Parser, T-J Yang, et al. arXiv:1902.05093.
  - Panoptic-DeepLab, B. Cheng, et al. In CVPR, 2020.

semantic_maps: An int32 Tensor with shape `[batch, height, width]` whose value
  indicates the predicted semantic label of each pixel.
instance_maps: An int32 Tensor with shape `[batch, height, width]` whose value
  indicates the predicted instance label of each pixel.
thing_ids: An int32 Tensor with shape `[num_thing_ids]` whose value refers to
  the semantic ids of the thing classes.
label_divisor: An integer. The value used to combine the semantic and instance
  map to generate the parsing map. In particular, the value of a pixel in the
  parsing map is equal to its corresponding semantic label times label_divisor
  plus instance label (i.e., semantic_label * label_divisor + instance_label).
stuff_area_limit: An integer. Predicted stuff segments whose areas are smaller
  than this threshold are assigned to VOID label.
void_label: An integer, specifying the VOID label.
parsing_maps: An int32 Tensor with shape `[batch, height, width]` whose value
  indicates the merged semantic and instance label of each pixel.
)doc");

}  // namespace deeplab2
}  // namespace deeplab
}  // namespace tensorflow_models
