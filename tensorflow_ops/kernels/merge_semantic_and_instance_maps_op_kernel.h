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

#ifndef DEEPLAB2_MERGE_SEMANTIC_AND_INSTANCE_MAPS_OP_KERNEL_H_
#define DEEPLAB2_MERGE_SEMANTIC_AND_INSTANCE_MAPS_OP_KERNEL_H_
#include <stdint.h>

#include <unordered_set>

#include /*third_party*/"tensorflow/core/framework/numeric_types.h"
#include /*third_party*/"tensorflow/core/framework/op_kernel.h"
#include /*third_party*/"tensorflow/core/framework/tensor.h"
#include /*third_party*/"tensorflow/core/framework/tensor_types.h"

namespace tensorflow_models {
namespace deeplab {
namespace deeplab2 {
namespace functor {

template <typename Device>
struct MergeSemanticAndInstanceMaps {
  // Functor that merges semantic and instance maps.
  void operator()(
      const Device& d,
      typename tensorflow::TTypes<int32_t, 3>::ConstTensor semantic_maps,
      typename tensorflow::TTypes<int32_t, 3>::ConstTensor instance_maps,
      const std::unordered_set<int32_t>& thing_ids_set, int label_divisor,
      int stuff_area_limit, int void_label,
      typename tensorflow::TTypes<int32_t, 3>::Tensor parsing_maps);
};

// Helper method to convert a list of thing IDs into hashset.
template <typename Device>
std::unordered_set<int32_t> Convert1DInt32TensorToSet(
    const Device& d, const tensorflow::Tensor& tensor);

}  // namespace functor
}  // namespace deeplab2
}  // namespace deeplab
}  // namespace tensorflow_models

#endif  // DEEPLAB2_MERGE_SEMANTIC_AND_INSTANCE_MAPS_OP_KERNEL_H_
