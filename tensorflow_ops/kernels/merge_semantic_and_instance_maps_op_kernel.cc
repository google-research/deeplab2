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

#include <cstdint>
#define EIGEN_USE_THREADS

#define _USE_MATH_DEFINES

#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include /*third_party*/"tensorflow/core/framework/op_kernel.h"
#include /*third_party*/"tensorflow/core/framework/register_types.h"
#include /*third_party*/"tensorflow/core/framework/tensor.h"
#include /*third_party*/"tensorflow/core/framework/tensor_shape.h"
#include /*third_party*/"tensorflow/core/framework/types.h"
#include /*third_party*/"tensorflow/core/lib/core/errors.h"
#include /*third_party*/"tensorflow/core/lib/core/status.h"
#include /*third_party*/"tensorflow/core/platform/logging.h"
#include /*third_party*/"merge_semantic_and_instance_maps_op_kernel.h" // local headers

namespace tensorflow_models {
namespace deeplab {
namespace deeplab2 {

namespace {

using tensorflow::Tensor;
using tensorflow::TensorShape;
using tensorflow::TTypes;
using tensorflow::errors::InvalidArgument;

}  // namespace

namespace functor {

// This function merges the semantic segmentation and class-agnostic
// instance segmentation to form the panoptic segmentation. In particular,
// the class label of each instance mask is inferred from the majority
// votes from the corresponding pixels in the semantic segmentation. This
// operation is first poposed in the DeeperLab paper and adopted by the
// Panoptic-DeepLab.
// - DeeperLab: Single-Shot Image Parser, T-J Yang, et al. arXiv:1902.05093.
// - Panoptic-DeepLab, B. Cheng, et al. In CVPR, 2020.
// Specialization of MergeSemanticAndInstanceMaps< for CPU.
template <>
void MergeSemanticAndInstanceMaps<Eigen::ThreadPoolDevice>::operator()(
    const Eigen::ThreadPoolDevice& d,
    typename TTypes<int32_t, 3>::ConstTensor semantic_maps,
    typename TTypes<int32_t, 3>::ConstTensor instance_maps,
    const std::unordered_set<int32_t>& thing_ids_set, int label_divisor,
    int stuff_area_limit, int void_label,
    typename TTypes<int32_t, 3>::Tensor parsing_maps) {
  const int num_batches = semantic_maps.dimension(0);
  const int height = semantic_maps.dimension(1);
  const int width = semantic_maps.dimension(2);

  for (int b = 0; b < num_batches; ++b) {
    // A vector to keep track of which pixels are predicted as `thing` or
    // `stuff` class.
    std::vector<bool> is_thing(height * width, true);

    // For each instance, find its corresponding histogram of semantic labels.
    // Suppose car label = 2 and road label = 5, and predicted instance 3 has
    // 5 pixels predicted as car and 20 pixels predicted as road. Then,
    // instance_id_to_semantic_histogram[3][2] = 5 and
    // instance_id_to_semantic_histogram[3][5] = 20.
    using InstanceIdType = int32_t;
    using SemanticLabelType = int32_t;
    using CountsType = int32_t;
    std::unordered_map<InstanceIdType,
                       std::unordered_map<SemanticLabelType, CountsType>>
        instance_id_to_semantic_histogram;
    // A map from stuff label to area.
    std::unordered_map<SemanticLabelType, CountsType> stuff_label_to_area;
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int semantic_val = semantic_maps(b, h, w);
        if (thing_ids_set.find(semantic_val) == thing_ids_set.end()) {
          // Skip if it is `stuff`.
          is_thing[w + width * h] = false;
          ++stuff_label_to_area[semantic_val];
          continue;
        }
        const int instance_val = instance_maps(b, h, w);
        ++instance_id_to_semantic_histogram[instance_val][semantic_val];
      }
    }
    // Keep track of how many instances for each semantic_label.
    std::unordered_map<SemanticLabelType, CountsType>
        semantic_label_to_instance_counts;
    // Find the new semantic label and instance id for each instance. We use
    // majority vote to find the new semantic label while reorder the instance
    // id in the following way. In the original instance map, every instance
    // has a different instance id. In the new instance map, every instance
    // `in the same semantic class` should have a different id, but instances
    // `in different semantic classes` can have the same instance id. This
    // reduces the maximum instance label value and avoids the problem of
    // combining the two maps with the label_divisor.
    std::unordered_map<InstanceIdType,
                       std::pair<SemanticLabelType, InstanceIdType>>
        instance_id_to_new_semantic_label_and_instance_id;
    for (const auto& instance_to_histogram :
         instance_id_to_semantic_histogram) {
      const int instance_val = instance_to_histogram.first;
      const std::unordered_map<SemanticLabelType, CountsType>
          semantic_histogram = instance_to_histogram.second;
      int semantic_label = -1;
      int max_count = 0;
      // Find the majority semantic label.
      for (const auto& semantic_to_count : semantic_histogram) {
        // Break ties deterministically by select the smaller semantic label.
        if (semantic_to_count.second > max_count ||
            (semantic_to_count.second == max_count &&
             semantic_to_count.first < semantic_label)) {
          max_count = semantic_to_count.second;
          semantic_label = semantic_to_count.first;
        }
      }
      ++semantic_label_to_instance_counts[semantic_label];
      // For `thing` class, we set instance id starting from 1, while for
      // `stuff` class, we use instance id 0.
      instance_id_to_new_semantic_label_and_instance_id[instance_val] = {
          semantic_label, semantic_label_to_instance_counts[semantic_label]};
    }
    // Create a new semantic map by assigning the majority semantic label for
    // each instance.
    std::vector<SemanticLabelType> semantic_map(height * width);
    // Create a new instance map by assigning ordered instance id's.
    std::vector<InstanceIdType> instance_map(height * width);
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int pixel = w + width * h;
        if (is_thing[pixel]) {
          const int instance_val = instance_maps(b, h, w);
          // Assign the majority semantic vote in the new semantic map, and
          // reorder the instance id in the new instance map.
          std::tie(semantic_map[pixel], instance_map[pixel]) =
              instance_id_to_new_semantic_label_and_instance_id[instance_val];
        } else {
          // If current pixel belongs to `stuff` class, keep the same semantic
          // label in the new semantic map. We also check if its area is
          // smaller than the stuff_area_limit_ or not. If true, we re-assign
          // the segment with void_label_.
          const int semantic_val = semantic_maps(b, h, w);
          if (stuff_area_limit > 0 &&
              stuff_label_to_area[semantic_val] <= stuff_area_limit) {
            semantic_map[pixel] = void_label;
          } else {
            semantic_map[pixel] = semantic_val;
          }
          // If current pixel belongs to `stuff` class, assign 0 in the new
          // instance map.
          instance_map[pixel] = 0;
        }
      }
    }
    // Merge those semantic map and instance map.
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
        const int pixel = w + width * h;
        parsing_maps(b, h, w) =
            semantic_map[pixel] * label_divisor + instance_map[pixel];
      }
    }
  }
}

template <>
std::unordered_set<int32_t> Convert1DInt32TensorToSet(
    const Eigen::ThreadPoolDevice& d, const Tensor& tensor) {
  std::unordered_set<int32_t> target_set;
  const int n_vals = tensor.dim_size(0);
  typename TTypes<int32_t, 1>::ConstTensor tensor_data =
      tensor.tensor<int32_t, 1>();
  for (int i = 0; i < n_vals; i++) {
    target_set.insert(tensor_data(i));
  }

  return target_set;
}

}  // namespace functor

template <typename Device>
class MergeSemanticAndInstanceMapsOp : public tensorflow::OpKernel {
 public:
  explicit MergeSemanticAndInstanceMapsOp(
      tensorflow::OpKernelConstruction* context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("label_divisor", &label_divisor_));
    OP_REQUIRES(context, label_divisor_ > 0,
                InvalidArgument("Label divisor must be positive."));
    OP_REQUIRES_OK(context,
                   context->GetAttr("stuff_area_limit", &stuff_area_limit_));
    OP_REQUIRES(context, stuff_area_limit_ >= 0,
                InvalidArgument("Stuff area limit must be non-negative."));
    OP_REQUIRES_OK(context, context->GetAttr("void_label", &void_label_));
    OP_REQUIRES(context, void_label_ >= 0,
                InvalidArgument("Void label must be non-negative."));
  }

  void Compute(tensorflow::OpKernelContext* context) override {
    // Extract the inputs.
    const Tensor& semantic_maps = context->input(0);
    const Tensor& instance_maps = context->input(1);
    const Tensor& thing_ids_tensor = context->input(2);

    // Convert thing_ids_tensor into a set.
    std::unordered_set<int32_t> thing_ids_set =
        functor::Convert1DInt32TensorToSet(context->eigen_device<Device>(),
                                           thing_ids_tensor);

    // Extract the constants.
    const int batch = semantic_maps.dim_size(0);
    const int height = semantic_maps.dim_size(1);
    const int width = semantic_maps.dim_size(2);

    // Check input shapes.
    OP_REQUIRES(context,
                instance_maps.dim_size(0) == batch &&
                    instance_maps.dim_size(1) == height &&
                    instance_maps.dim_size(2) == width,
                InvalidArgument(
                    "Expect semantic and instance maps have the same shape.",
                    instance_maps.shape().DebugString()));

    Tensor* parsing_maps = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(
                       0, TensorShape({batch, height, width}), &parsing_maps));

    functor::MergeSemanticAndInstanceMaps<Device>()(
        context->eigen_device<Device>(), semantic_maps.tensor<int32_t, 3>(),
        instance_maps.tensor<int32_t, 3>(), thing_ids_set, label_divisor_,
        stuff_area_limit_, void_label_, parsing_maps->tensor<int32_t, 3>());
  }

 private:
  // Label divisor, the value used to combine the semantic and instance map to
  // generate the parsing map.
  int label_divisor_;

  // Stuff area limit is used to remove predicted stuff segments whose area are
  // smaller than it.
  int stuff_area_limit_;

  // Removed predicted stuff segments are re-assigned with void label.
  int void_label_;
};

REGISTER_KERNEL_BUILDER(
    Name("MergeSemanticAndInstanceMaps").Device(tensorflow::DEVICE_CPU),
    MergeSemanticAndInstanceMapsOp<Eigen::ThreadPoolDevice>);

#ifdef GOOGLE_CUDA
REGISTER_KERNEL_BUILDER(
    Name("MergeSemanticAndInstanceMaps").Device(tensorflow::DEVICE_GPU),
    MergeSemanticAndInstanceMapsOp<Eigen::GpuDevice>)
#endif  // GOOGLE_CUDA

}  // namespace deeplab2
}  // namespace deeplab
}  // namespace tensorflow_models
