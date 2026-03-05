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

#ifndef CONTAINERS_DEMO_CLUSTER_CC_FAKE_DATA_READ_WRITE_FFI_H_
#define CONTAINERS_DEMO_CLUSTER_CC_FAKE_DATA_READ_WRITE_FFI_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle to FakeDataReadWriteService and its gRPC server
typedef struct FakeDataReadWriteHandle FakeDataReadWriteHandle;

// Creates a new FakeDataReadWriteService instance.
// Returns NULL on failure.
FakeDataReadWriteHandle* fake_data_rw_create();

// Destroys the FakeDataReadWriteService instance and stops the server.
void fake_data_rw_destroy(FakeDataReadWriteHandle* handle);

// Starts the gRPC server on the given address (e.g., "[::1]:0" for random port).
// Returns the actual port number on success, or -1 on failure.
int fake_data_rw_start_server(FakeDataReadWriteHandle* handle,
                               const char* address);

// Gets the server address after starting. Writes to out_buf and returns length.
// Returns 0 if server not started or buffer too small.
size_t fake_data_rw_get_server_address(FakeDataReadWriteHandle* handle,
                                        char* out_buf, size_t buf_len);

// Stores an encrypted message for the given URI.
// blob_id can be NULL (will use default).
// Returns 0 on success, non-zero on failure.
int fake_data_rw_store_encrypted(FakeDataReadWriteHandle* handle,
                                  const char* uri,
                                  const uint8_t* message, size_t message_len,
                                  const uint8_t* blob_id, size_t blob_id_len);

// Stores a plaintext message for the given URI.
// Returns 0 on success, non-zero on failure.
int fake_data_rw_store_plaintext(FakeDataReadWriteHandle* handle,
                                  const char* uri,
                                  const uint8_t* message, size_t message_len);

// Gets the input public key. Writes to out_buf and returns actual length.
// Returns 0 if buffer too small.
size_t fake_data_rw_get_input_public_key(FakeDataReadWriteHandle* handle,
                                          uint8_t* out_buf, size_t buf_len);

// Gets the input private key. Writes to out_buf and returns actual length.
// Returns 0 if buffer too small.
size_t fake_data_rw_get_input_private_key(FakeDataReadWriteHandle* handle,
                                           uint8_t* out_buf, size_t buf_len);

// Gets the result public key. Writes to out_buf and returns actual length.
// Returns 0 if buffer too small.
size_t fake_data_rw_get_result_public_key(FakeDataReadWriteHandle* handle,
                                           uint8_t* out_buf, size_t buf_len);

// Gets the result private key. Writes to out_buf and returns actual length.
// Returns 0 if buffer too small.
size_t fake_data_rw_get_result_private_key(FakeDataReadWriteHandle* handle,
                                            uint8_t* out_buf, size_t buf_len);

// Gets the released data for a given key. Writes to out_buf and returns length.
// Returns 0 if key not found or buffer too small.
size_t fake_data_rw_get_released_data(FakeDataReadWriteHandle* handle,
                                       const char* key,
                                       uint8_t* out_buf, size_t buf_len);

// Gets the number of released data entries.
size_t fake_data_rw_get_released_data_count(FakeDataReadWriteHandle* handle);

// Gets the key at the given index in the released data map.
// Returns the length of the key written to out_buf, or 0 if index out of bounds.
size_t fake_data_rw_get_released_data_key_at(FakeDataReadWriteHandle* handle,
                                              size_t index,
                                              char* out_buf, size_t buf_len);

#ifdef __cplusplus
}
#endif

#endif  // CONTAINERS_DEMO_CLUSTER_CC_FAKE_DATA_READ_WRITE_FFI_H_
