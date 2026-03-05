// Copyright 2025 Google LLC.
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

// Local copy of generate_checkpoint.h from
// program_containers/program_executor_tee/program_context/cc/,
// extended with float32 and multi-tensor checkpoint support for CIFAR-10.

#ifndef CONTAINERS_DEMO_CLUSTER_CC_GENERATE_CHECKPOINT_H_
#define CONTAINERS_DEMO_CLUSTER_CC_GENERATE_CHECKPOINT_H_

#include <cstdint>
#include <string>
#include <vector>

namespace demo_cluster {

// Returns an unencrypted federated compute checkpoint that stores a tensor with
// the provided integer input_values at the key tensor_name.
std::string BuildClientCheckpointFromInts(std::vector<int> input_values,
                                          std::string tensor_name);

// Returns an unencrypted federated compute checkpoint that stores a tensor with
// the provided string input_values at the key tensor_name.
std::string BuildClientCheckpointFromStrings(
    std::vector<std::string> input_values, std::string tensor_name);

// Returns an unencrypted federated compute checkpoint that stores a float32
// tensor with the provided input_values and shape at the key tensor_name.
std::string BuildClientCheckpointFromFloats(std::vector<float> input_values,
                                            std::vector<int32_t> shape,
                                            std::string tensor_name);

// Returns an unencrypted federated compute checkpoint that stores two tensors:
// a float32 tensor and an int32 tensor, each with their own shape and name.
// This is used for CIFAR-10 data where images (float32) and labels (int32)
// are stored together in a single checkpoint.
std::string BuildClientCheckpointFromFloatsAndInts(
    std::vector<float> float_values, std::vector<int32_t> float_shape,
    std::string float_tensor_name, std::vector<int32_t> int_values,
    std::vector<int32_t> int_shape, std::string int_tensor_name);

}  // namespace demo_cluster

#endif  // CONTAINERS_DEMO_CLUSTER_CC_GENERATE_CHECKPOINT_H_