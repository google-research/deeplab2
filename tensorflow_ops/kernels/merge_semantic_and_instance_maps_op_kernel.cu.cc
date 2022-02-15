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
#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include <unordered_map>
#include <unordered_set>
#include <vector>

#include /*third_party*/"tensorflow/core/framework/op_kernel.h"
#include /*third_party*/"tensorflow/core/framework/register_types.h"
#include /*third_party*/"tensorflow/core/framework/tensor.h"
#include /*third_party*/"tensorflow/core/framework/tensor_shape.h"
#include /*third_party*/"tensorflow/core/framework/types.h"
#include /*third_party*/"tensorflow/core/util/gpu_kernel_helper.h"
#include /*third_party*/"merge_semantic_and_instance_maps_op_kernel.h" // local headers

namespace tensorflow_models {
namespace deeplab {
namespace deeplab2 {

namespace functor {

namespace {

using ::tensorflow::CudaGridRangeX;
using ::tensorflow::GetGpuLaunchConfig;
using ::tensorflow::GpuLaunchConfig;
using ::tensorflow::Tensor;
using ::tensorflow::TTypes;

using GPUDevice = ::Eigen::GpuDevice;

// Maximum number of instances and semantic classes. We default to
// 1024 and 256, respectively. Increase the values, if your dataset
// contains more instances per image or more semantic classes.
constexpr int32_t kMaxNumInstance = 1024;
constexpr int32_t kMaxNumSemantic = 256;

// CUDA kernel that initializes memory with a constant value.
template <typename T>
__global__ void SetToValue(const int num_threads, const T value, T* x) {
  for (int idx : CudaGridRangeX(num_threads)) {
    x[idx] = value;
  }
}

// CUDA kernel that goes over each pixel, and collects the following stats:
// 1. Whether this pixel belongs to "thing" class.
// 2. Semantic label count inside each instance.
// 3. Total pixel area of each "stuff" class.
// Size of each GPU array:
//   semantic_data: [height * width]
//   instance_data: [height * width]
//   is_thing_per_semantic_id: [kMaxNumSemantic]
//   is_thing_per_pixel: [height * width]
//   semantic_count_per_instance: [kMaxNumInstance * kMaxNumSemantic]
//   stuff_area: [kMaxNumSemantic]
__global__ void CollectPixelStats(const int num_threads,
                                  const int32_t* semantic_data,
                                  const int32_t* instance_data,
                                  const bool* is_thing_per_semantic_id,
                                  bool* is_thing_per_pixel,
                                  int32_t* semantic_count_per_instance,
                                  int32_t* stuff_area) {
  for (int idx : CudaGridRangeX(num_threads)) {
    const int32_t semantic_label =
        std::min(semantic_data[idx], kMaxNumSemantic - 1);
    const int32_t instance_label =
        std::min(instance_data[idx], kMaxNumInstance - 1);
    const bool is_thing = is_thing_per_semantic_id[semantic_label];
    is_thing_per_pixel[idx] = is_thing;

    const int offset = instance_label * kMaxNumSemantic + semantic_label;
    if (is_thing) {
      tensorflow::CudaAtomicAdd(semantic_count_per_instance + offset, 1);
    } else {
      tensorflow::CudaAtomicAdd(stuff_area + semantic_label, 1);
    }
  }
}

// CUDA kernel that merges semantic and instance prediction into panoptic map.
// Merging rules:
// 1. For "thing" class, its instance label will be reordered, and its semantic
//    label depends on major semantic label inside this instance.
// 2. For "stuff" class, its instance label is 0, and semantic label will be
//    a) void, if stuff area is small, and b) original semantic label.
// Size of each GPU array:
//   semantic_data: [height * width]
//   instance_data: [height * width]
//   is_thing_per_semantic_id: [kMaxNumSemantic]
//   is_thing_per_pixel: [height * width]
//   stuff_area: [kMaxNumSemantic]
//   labels_per_instance: [kMaxNumInstance * 2]
//   parsing_maps: [height * width]
__global__ void MergePredictions(
    const int num_threads, const int32_t* semantic_data,
    const int32_t* instance_data, const bool* is_thing_per_pixel,
    const int32_t* stuff_area, const int32_t* labels_per_instance,
    const int32_t stuff_area_limit, const int32_t label_divisor,
    const int32_t void_label, int32_t* parsing_maps) {
  for (int idx : CudaGridRangeX(num_threads)) {
    const int32_t semantic_label =
        std::min(semantic_data[idx], kMaxNumSemantic - 1);
    const int32_t instance_label =
        std::min(instance_data[idx], kMaxNumInstance - 1);
    const int32_t is_thing = static_cast<int32_t>(is_thing_per_pixel[idx]);

    const int32_t semantic_label_if_is_thing =
        labels_per_instance[instance_label * 2];
    const int32_t instance_label_if_is_thing =
        labels_per_instance[instance_label * 2 + 1];
    const int32_t panoptic_label_if_is_thing =
        semantic_label_if_is_thing * label_divisor + instance_label_if_is_thing;

    const int32_t is_void = static_cast<int32_t>(
        stuff_area_limit > 0 && stuff_area[semantic_label] <= stuff_area_limit);
    const int32_t semantic_label_if_is_stuff =
        is_void * void_label + (1 - is_void) * semantic_label;

    parsing_maps[idx] =
        is_thing * panoptic_label_if_is_thing +
        (1 - is_thing) * (semantic_label_if_is_stuff * label_divisor);
  }
}

// Generates semantic and instance label for each predicted instance.
// Size of each GPU array:
//   semantic_count_per_instance: [kMaxNumInstance * kMaxNumSemantic]
//   labels_per_instance: [kMaxNumInstance * 2]
void CreateLabelsPerInstance(const GPUDevice& d,
                             const int32_t* semantic_count_per_instance,
                             int32_t* labels_per_instance) {
  std::vector<int32_t> semantic_count_per_instance_host(kMaxNumInstance *
                                                        kMaxNumSemantic);
  d.memcpyDeviceToHost(semantic_count_per_instance_host.data(),
                       semantic_count_per_instance,
                       kMaxNumInstance * kMaxNumSemantic * sizeof(int32_t));

  // A flat 2D array with shape [kMaxNumInstance, 2], where each row
  // represents (new semantic label, new instance label) for each instance.
  std::vector<int32_t> labels_per_instance_host(kMaxNumInstance * 2);

  // Map semantic_label -> largest instance label of this semantic class.
  std::unordered_map<int32_t, int32_t> instance_count_per_semantic_class;
  for (int i = 0; i < kMaxNumInstance; ++i) {
    int max_pixel_count = 0;
    int max_semantic_label = -1;
    for (int j = 0; j < kMaxNumSemantic; ++j) {
      const int current_count =
          semantic_count_per_instance_host[i * kMaxNumSemantic + j];
      if (current_count > max_pixel_count) {
        max_semantic_label = j;
        max_pixel_count = current_count;
      }
    }

    labels_per_instance_host[2 * i] = std::max(0, max_semantic_label);
    if (max_semantic_label >= 0) {
      labels_per_instance_host[2 * i + 1] =
          ++instance_count_per_semantic_class[max_semantic_label];
    } else {
      labels_per_instance_host[2 * i + 1] = 0;
    }
  }

  d.memcpyHostToDevice(labels_per_instance, labels_per_instance_host.data(),
                       kMaxNumInstance * 2 * sizeof(int32_t));
}

}  // namespace

// Specialization of Convert1DInt32TensorToSet for GPU.
template <>
std::unordered_set<int32_t> Convert1DInt32TensorToSet(const GPUDevice& d,
                                                      const Tensor& tensor) {
  const int n_vals = tensor.dim_size(0);
  std::vector<int32_t> host_buffer(n_vals);
  d.memcpyDeviceToHost(host_buffer.data(), tensor.tensor<int32_t, 1>().data(),
                       n_vals * sizeof(int32_t));

  return std::unordered_set<int32_t>(host_buffer.begin(), host_buffer.end());
}

// This function merges the semantic segmentation and class-agnostic
// instance segmentation to form the panoptic segmentation. In particular,
// the class label of each instance mask is inferred from the majority
// votes from the corresponding pixels in the semantic segmentation. This
// operation is first poposed in the DeeperLab paper and adopted by the
// Panoptic-DeepLab.
// - DeeperLab: Single-Shot Image Parser, T-J Yang, et al. arXiv:1902.05093.
// - Panoptic-DeepLab, B. Cheng, et al. In CVPR, 2020.
// Specialization of MergeSemanticAndInstanceMaps for GPU.
template <>
void MergeSemanticAndInstanceMaps<GPUDevice>::operator()(
    const GPUDevice& d, typename TTypes<int32_t, 3>::ConstTensor semantic_maps,
    typename TTypes<int32_t, 3>::ConstTensor instance_maps,
    const std::unordered_set<int32_t>& thing_ids_set, int label_divisor,
    int stuff_area_limit, int void_label,
    typename TTypes<int32_t, 3>::Tensor parsing_maps) {
  const int num_batches = semantic_maps.dimension(0);
  const int height = semantic_maps.dimension(1);
  const int width = semantic_maps.dimension(2);

  // Allocate memory on host, which tells each semantic class is "thing" or not.
  bool is_thing_per_semantic_id[kMaxNumSemantic];
  for (int i = 0; i < kMaxNumSemantic; ++i) {
    is_thing_per_semantic_id[i] =
        (thing_ids_set.find(i) != thing_ids_set.end());
  }
  bool* is_thing_per_semantic_id_device =
      reinterpret_cast<bool*>(d.allocate_temp(kMaxNumSemantic * sizeof(bool)));
  d.memcpyHostToDevice(is_thing_per_semantic_id_device,
                       is_thing_per_semantic_id,
                       kMaxNumSemantic * sizeof(bool));

  // Allocate scratch memories on device.
  bool* is_thing_per_pixel_device =
      reinterpret_cast<bool*>(d.allocate_temp(height * width * sizeof(bool)));
  int32_t* semantic_count_per_instance_device = reinterpret_cast<int32_t*>(
      d.allocate_temp(kMaxNumInstance * kMaxNumSemantic * sizeof(int32_t)));
  int32_t* stuff_area_device = reinterpret_cast<int32_t*>(
      d.allocate_temp(kMaxNumSemantic * sizeof(int32_t)));
  int32_t* labels_per_instance_device = reinterpret_cast<int32_t*>(
      d.allocate_temp(kMaxNumInstance * 2 * sizeof(int32_t)));

  GpuLaunchConfig config;
  int total_count = 0;
  for (int b = 0; b < num_batches; ++b) {
    const int batch_offset = b * height * width;
    // Initialize memories that hold counters.
    total_count = kMaxNumInstance * kMaxNumSemantic;
    config = GetGpuLaunchConfig(total_count, d);
    SetToValue<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, 0, semantic_count_per_instance_device);

    total_count = kMaxNumSemantic;
    config = GetGpuLaunchConfig(total_count, d);
    SetToValue<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, 0, stuff_area_device);

    // Step 1: Collect semantic and instance mask stats. Done on GPU.
    total_count = height * width;
    config = GetGpuLaunchConfig(total_count, d);
    CollectPixelStats<<<config.block_count, config.thread_per_block, 0,
                        d.stream()>>>(
        config.virtual_thread_count, semantic_maps.data() + batch_offset,
        instance_maps.data() + batch_offset, is_thing_per_semantic_id_device,
        is_thing_per_pixel_device, semantic_count_per_instance_device,
        stuff_area_device);

    // Step 2: Loop over instance, find major "thing" semantic label, and
    //         reorder instance IDs to share same ID with different thing class.
    //         This process now runs on CPU.
    CreateLabelsPerInstance(d, semantic_count_per_instance_device,
                            labels_per_instance_device);

    // Step 3: Create panoptic prediction.
    total_count = width * height;
    config = GetGpuLaunchConfig(total_count, d);
    MergePredictions<<<config.block_count, config.thread_per_block, 0,
                       d.stream()>>>(
        config.virtual_thread_count, semantic_maps.data() + batch_offset,
        instance_maps.data() + batch_offset, is_thing_per_pixel_device,
        stuff_area_device, labels_per_instance_device, stuff_area_limit,
        label_divisor, void_label, parsing_maps.data() + batch_offset);
  }

  // Free all temp memories.
  d.deallocate_temp(is_thing_per_semantic_id_device);
  d.deallocate_temp(is_thing_per_pixel_device);
  d.deallocate_temp(semantic_count_per_instance_device);
  d.deallocate_temp(stuff_area_device);
  d.deallocate_temp(labels_per_instance_device);
}

}  // namespace functor
}  // namespace deeplab2
}  // namespace deeplab
}  // namespace tensorflow_models

#endif  // GOOGLE_CUDA
