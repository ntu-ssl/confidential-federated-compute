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

#include "cc/checkpoint_ffi.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cc/generate_checkpoint.h"

extern "C" {

size_t checkpoint_build_from_ints(const int* values, size_t values_len,
                                   const char* tensor_name, uint8_t* out_buf,
                                   size_t buf_len) {
  if (!values || !tensor_name || !out_buf) {
    return 0;
  }

  std::vector<int> input_values(values, values + values_len);
  std::string checkpoint =
      demo_cluster::BuildClientCheckpointFromInts(input_values, tensor_name);

  if (buf_len < checkpoint.size()) {
    return 0;
  }

  memcpy(out_buf, checkpoint.data(), checkpoint.size());
  return checkpoint.size();
}

size_t checkpoint_build_from_floats(const float* values, size_t values_len,
                                    const int32_t* shape, size_t shape_len,
                                    const char* tensor_name, uint8_t* out_buf,
                                    size_t buf_len) {
  if (!values || !shape || !tensor_name || !out_buf) {
    return 0;
  }

  std::vector<float> input_values(values, values + values_len);
  std::vector<int32_t> shape_vec(shape, shape + shape_len);
  std::string checkpoint =
      demo_cluster::BuildClientCheckpointFromFloats(
          input_values, shape_vec, tensor_name);

  if (buf_len < checkpoint.size()) {
    return 0;
  }

  memcpy(out_buf, checkpoint.data(), checkpoint.size());
  return checkpoint.size();
}

size_t checkpoint_build_float_int_tensors(
    const float* float_data, size_t float_data_len,
    const int32_t* float_shape, size_t float_shape_len,
    const char* float_tensor_name,
    const int32_t* int_data, size_t int_data_len,
    const int32_t* int_shape, size_t int_shape_len,
    const char* int_tensor_name,
    uint8_t* out_buf, size_t buf_len) {
  if (!float_data || !float_shape || !float_tensor_name ||
      !int_data || !int_shape || !int_tensor_name || !out_buf) {
    return 0;
  }

  std::vector<float> float_values(float_data, float_data + float_data_len);
  std::vector<int32_t> float_shape_vec(float_shape,
                                       float_shape + float_shape_len);
  std::vector<int32_t> int_values(int_data, int_data + int_data_len);
  std::vector<int32_t> int_shape_vec(int_shape, int_shape + int_shape_len);

  std::string checkpoint =
      demo_cluster::BuildClientCheckpointFromFloatsAndInts(
          float_values, float_shape_vec, float_tensor_name,
          int_values, int_shape_vec, int_tensor_name);

  if (buf_len < checkpoint.size()) {
    return 0;
  }

  memcpy(out_buf, checkpoint.data(), checkpoint.size());
  return checkpoint.size();
}

}  // extern "C"