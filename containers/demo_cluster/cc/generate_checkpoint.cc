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

// Local copy of generate_checkpoint.cc from
// program_containers/program_executor_tee/program_context/cc/,
// extended with float32 and multi-tensor checkpoint support for CIFAR-10.

#include "cc/generate_checkpoint.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_string_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/mutable_vector_data.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor.h"
#include "tensorflow_federated/cc/core/impl/aggregation/core/tensor_shape.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/checkpoint_builder.h"
#include "tensorflow_federated/cc/core/impl/aggregation/protocol/federated_compute_checkpoint_builder.h"

namespace demo_cluster {

using ::tensorflow_federated::aggregation::CheckpointBuilder;
using ::tensorflow_federated::aggregation::DataType;
using ::tensorflow_federated::aggregation::
    FederatedComputeCheckpointBuilderFactory;
using ::tensorflow_federated::aggregation::MutableStringData;
using ::tensorflow_federated::aggregation::MutableVectorData;
using ::tensorflow_federated::aggregation::Tensor;
using ::tensorflow_federated::aggregation::TensorShape;

namespace {

std::string BuildClientCheckpointFromTensor(Tensor tensor,
                                            std::string tensor_name) {
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();
  CHECK_OK(ckpt_builder->Add(tensor_name, tensor));
  auto checkpoint = ckpt_builder->Build();
  return std::string(*checkpoint);
}

}  // namespace

std::string BuildClientCheckpointFromInts(std::vector<int> input_values,
                                          std::string tensor_name) {
  absl::StatusOr<Tensor> t =
      Tensor::Create(DataType::DT_INT32,
                     TensorShape({static_cast<int32_t>(input_values.size())}),
                     std::make_unique<MutableVectorData<int32_t>>(
                         input_values.begin(), input_values.end()));
  CHECK_OK(t);
  return BuildClientCheckpointFromTensor(std::move(*t), tensor_name);
}

std::string BuildClientCheckpointFromStrings(
    std::vector<std::string> input_values, std::string tensor_name) {
  auto data = std::make_unique<MutableStringData>(input_values.size());
  for (std::string& value : input_values) {
    data->Add(std::move(value));
  }
  absl::StatusOr<Tensor> t =
      Tensor::Create(DataType::DT_STRING,
                     TensorShape({static_cast<int32_t>(input_values.size())}),
                     std::move(data));
  CHECK_OK(t);
  return BuildClientCheckpointFromTensor(std::move(*t), tensor_name);
}

std::string BuildClientCheckpointFromFloats(std::vector<float> input_values,
                                            std::vector<int32_t> shape,
                                            std::string tensor_name) {
  absl::StatusOr<Tensor> t =
      Tensor::Create(DataType::DT_FLOAT,
                     TensorShape(shape.begin(), shape.end()),
                     std::make_unique<MutableVectorData<float>>(
                         input_values.begin(), input_values.end()));
  CHECK_OK(t);
  return BuildClientCheckpointFromTensor(std::move(*t), tensor_name);
}

std::string BuildClientCheckpointFromFloatsAndInts(
    std::vector<float> float_values, std::vector<int32_t> float_shape,
    std::string float_tensor_name, std::vector<int32_t> int_values,
    std::vector<int32_t> int_shape, std::string int_tensor_name) {
  // Create float tensor
  absl::StatusOr<Tensor> float_tensor =
      Tensor::Create(DataType::DT_FLOAT,
                     TensorShape(float_shape.begin(), float_shape.end()),
                     std::make_unique<MutableVectorData<float>>(
                         float_values.begin(), float_values.end()));
  CHECK_OK(float_tensor);

  // Create int tensor
  absl::StatusOr<Tensor> int_tensor =
      Tensor::Create(DataType::DT_INT32,
                     TensorShape(int_shape.begin(), int_shape.end()),
                     std::make_unique<MutableVectorData<int32_t>>(
                         int_values.begin(), int_values.end()));
  CHECK_OK(int_tensor);

  // Build checkpoint with both tensors
  FederatedComputeCheckpointBuilderFactory builder_factory;
  std::unique_ptr<CheckpointBuilder> ckpt_builder = builder_factory.Create();
  CHECK_OK(ckpt_builder->Add(float_tensor_name, *float_tensor));
  CHECK_OK(ckpt_builder->Add(int_tensor_name, *int_tensor));
  auto checkpoint = ckpt_builder->Build();
  return std::string(*checkpoint);
}

}  // namespace demo_cluster