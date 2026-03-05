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

#ifndef CONTAINERS_DEMO_CLUSTER_CC_CHECKPOINT_FFI_H_
#define CONTAINERS_DEMO_CLUSTER_CC_CHECKPOINT_FFI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Builds a FederatedCompute checkpoint from integer values.
//
// Creates a checkpoint in FCv1 format with a single INT32 tensor.
//
// Args:
//   values: Pointer to array of int32 values
//   values_len: Number of values in the array
//   tensor_name: Name of the tensor (null-terminated string)
//   out_buf: Output buffer for the checkpoint data
//   buf_len: Size of the output buffer
//
// Returns:
//   Number of bytes written to out_buf, or 0 on error.
size_t checkpoint_build_from_ints(const int* values, size_t values_len,
                                   const char* tensor_name, uint8_t* out_buf,
                                   size_t buf_len);

// Builds a FederatedCompute checkpoint from float values with arbitrary shape.
//
// Creates a checkpoint in FCv1 format with a single FLOAT tensor.
//
// Args:
//   values: Pointer to array of float values
//   values_len: Number of float values (product of all shape dimensions)
//   shape: Pointer to array of dimension sizes
//   shape_len: Number of dimensions
//   tensor_name: Name of the tensor (null-terminated string)
//   out_buf: Output buffer for the checkpoint data
//   buf_len: Size of the output buffer
//
// Returns:
//   Number of bytes written to out_buf, or 0 on error.
size_t checkpoint_build_from_floats(const float* values, size_t values_len,
                                    const int32_t* shape, size_t shape_len,
                                    const char* tensor_name, uint8_t* out_buf,
                                    size_t buf_len);

// Builds a FederatedCompute checkpoint with two tensors: a float32 tensor
// and an int32 tensor. Used for CIFAR-10 data (images + labels).
//
// Args:
//   float_data: Pointer to array of float values
//   float_data_len: Number of float values
//   float_shape: Pointer to array of dimension sizes for the float tensor
//   float_shape_len: Number of dimensions
//   float_tensor_name: Name of the float tensor (null-terminated string)
//   int_data: Pointer to array of int32 values
//   int_data_len: Number of int values
//   int_shape: Pointer to array of dimension sizes for the int tensor
//   int_shape_len: Number of dimensions
//   int_tensor_name: Name of the int tensor (null-terminated string)
//   out_buf: Output buffer for the checkpoint data
//   buf_len: Size of the output buffer
//
// Returns:
//   Number of bytes written to out_buf, or 0 on error.
size_t checkpoint_build_float_int_tensors(
    const float* float_data, size_t float_data_len,
    const int32_t* float_shape, size_t float_shape_len,
    const char* float_tensor_name,
    const int32_t* int_data, size_t int_data_len,
    const int32_t* int_shape, size_t int_shape_len,
    const char* int_tensor_name,
    uint8_t* out_buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif  // CONTAINERS_DEMO_CLUSTER_CC_CHECKPOINT_FFI_H_
