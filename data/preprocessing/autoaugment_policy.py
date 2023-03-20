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

"""AutoAugment policy file.

This file contains found auto-augment policy.

Please cite or refer to the following papers for details:
- Ekin D Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, and Quoc V Le.
"Autoaugment: Learning augmentation policies from data." In CVPR, 2019.

- Ekin D Cubuk, Barret Zoph, Jonathon Shlens, and Quoc V Le.
"Randaugment: Practical automated data augmentation with a reduced search
space." In CVPR, 2020.
"""

# Reduced augmentation operation space.
augmentation_reduced_operations = (
    'AutoContrast', 'Equalize', 'Invert', 'Posterize',
    'Solarize', 'Color', 'Contrast', 'Brightness', 'Sharpness')

augmentation_probabilities = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def convert_policy(policy,
                   search_space=augmentation_reduced_operations,
                   probability_scale=1.0,
                   magnitude_scale=1):
  """Converts policy from a list of numbers."""
  if len(policy) % 6:
    raise ValueError('Policy length must be a multiple of 6.')
  num_policies = len(policy) // 6
  policy_list = [[] for _ in range(num_policies)]
  for n in range(num_policies):
    for i in range(2):
      operation_id, prob_id, magnitude = (
          policy[6 * n + i * 3 : 6 * n + (i + 1) * 3])
      policy_name = search_space[operation_id]
      policy_prob = (
          augmentation_probabilities[prob_id] * probability_scale)
      policy_list[n].append((policy_name,
                             policy_prob,
                             magnitude * magnitude_scale))
  return policy_list


simple_classification_policy = [8, 2, 7, 7, 1, 10,
                                1, 0, 9, 6, 1, 10,
                                8, 1, 9, 5, 1, 9,
                                4, 1, 7, 1, 3, 9,
                                8, 1, 1, 1, 1, 7]

# All available policies.
available_policies = {
    'simple_classification_policy_magnitude_scale_0.2': convert_policy(
        simple_classification_policy,
        augmentation_reduced_operations,
        magnitude_scale=0.2),
    'simple_classification_policy': convert_policy(
        simple_classification_policy,
        augmentation_reduced_operations,
        magnitude_scale=1),
}
