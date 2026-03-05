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

#include "cc/fake_data_read_write_ffi.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "grpcpp/grpcpp.h"
#include "program_executor_tee/program_context/cc/fake_data_read_write_service.h"

using confidential_federated_compute::program_executor_tee::
    FakeDataReadWriteService;

struct FakeDataReadWriteHandle {
  std::unique_ptr<FakeDataReadWriteService> service;
  std::unique_ptr<grpc::Server> server;
  std::string server_address;
  int server_port = -1;
};

extern "C" {

FakeDataReadWriteHandle* fake_data_rw_create() {
  auto handle = new FakeDataReadWriteHandle();
  handle->service = std::make_unique<FakeDataReadWriteService>();
  return handle;
}

void fake_data_rw_destroy(FakeDataReadWriteHandle* handle) {
  if (handle) {
    if (handle->server) {
      handle->server->Shutdown();
    }
    delete handle;
  }
}

int fake_data_rw_start_server(FakeDataReadWriteHandle* handle,
                               const char* address) {
  if (!handle || !handle->service || !address) {
    return -1;
  }

  grpc::ServerBuilder builder;
  int selected_port = 0;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials(),
                           &selected_port);
  builder.RegisterService(handle->service.get());

  handle->server = builder.BuildAndStart();
  if (!handle->server) {
    return -1;
  }

  handle->server_port = selected_port;
  // Construct the actual address with the selected port
  std::string addr_str(address);
  size_t colon_pos = addr_str.rfind(':');
  if (colon_pos != std::string::npos) {
    handle->server_address =
        addr_str.substr(0, colon_pos + 1) + std::to_string(selected_port);
  } else {
    handle->server_address = addr_str + ":" + std::to_string(selected_port);
  }

  return selected_port;
}

size_t fake_data_rw_get_server_address(FakeDataReadWriteHandle* handle,
                                        char* out_buf, size_t buf_len) {
  if (!handle || handle->server_address.empty()) {
    return 0;
  }
  size_t len = handle->server_address.size();
  if (buf_len < len + 1) {
    return 0;
  }
  memcpy(out_buf, handle->server_address.c_str(), len);
  out_buf[len] = '\0';
  return len;
}

int fake_data_rw_store_encrypted(FakeDataReadWriteHandle* handle,
                                  const char* uri, const uint8_t* message,
                                  size_t message_len, const uint8_t* blob_id,
                                  size_t blob_id_len) {
  if (!handle || !handle->service || !uri || !message) {
    return -1;
  }

  std::optional<absl::string_view> blob_id_opt;
  if (blob_id && blob_id_len > 0) {
    blob_id_opt =
        absl::string_view(reinterpret_cast<const char*>(blob_id), blob_id_len);
  }

  absl::Status status = handle->service->StoreEncryptedMessageForKms(
      uri,
      absl::string_view(reinterpret_cast<const char*>(message), message_len),
      blob_id_opt);

  return status.ok() ? 0 : -1;
}

int fake_data_rw_store_plaintext(FakeDataReadWriteHandle* handle,
                                  const char* uri, const uint8_t* message,
                                  size_t message_len) {
  if (!handle || !handle->service || !uri || !message) {
    return -1;
  }

  absl::Status status = handle->service->StorePlaintextMessage(
      uri,
      absl::string_view(reinterpret_cast<const char*>(message), message_len));

  return status.ok() ? 0 : -1;
}

size_t fake_data_rw_get_input_public_key(FakeDataReadWriteHandle* handle,
                                          uint8_t* out_buf, size_t buf_len) {
  if (!handle || !handle->service) {
    return 0;
  }
  auto key_pair = handle->service->GetInputPublicPrivateKeyPair();
  const std::string& public_key = key_pair.first;
  if (buf_len < public_key.size()) {
    return 0;
  }
  memcpy(out_buf, public_key.data(), public_key.size());
  return public_key.size();
}

size_t fake_data_rw_get_input_private_key(FakeDataReadWriteHandle* handle,
                                           uint8_t* out_buf, size_t buf_len) {
  if (!handle || !handle->service) {
    return 0;
  }
  auto key_pair = handle->service->GetInputPublicPrivateKeyPair();
  const std::string& private_key = key_pair.second;
  if (buf_len < private_key.size()) {
    return 0;
  }
  memcpy(out_buf, private_key.data(), private_key.size());
  return private_key.size();
}

size_t fake_data_rw_get_result_public_key(FakeDataReadWriteHandle* handle,
                                           uint8_t* out_buf, size_t buf_len) {
  if (!handle || !handle->service) {
    return 0;
  }
  auto key_pair = handle->service->GetResultPublicPrivateKeyPair();
  const std::string& public_key = key_pair.first;
  if (buf_len < public_key.size()) {
    return 0;
  }
  memcpy(out_buf, public_key.data(), public_key.size());
  return public_key.size();
}

size_t fake_data_rw_get_result_private_key(FakeDataReadWriteHandle* handle,
                                            uint8_t* out_buf, size_t buf_len) {
  if (!handle || !handle->service) {
    return 0;
  }
  auto key_pair = handle->service->GetResultPublicPrivateKeyPair();
  const std::string& private_key = key_pair.second;
  if (buf_len < private_key.size()) {
    return 0;
  }
  memcpy(out_buf, private_key.data(), private_key.size());
  return private_key.size();
}

size_t fake_data_rw_get_released_data(FakeDataReadWriteHandle* handle,
                                       const char* key, uint8_t* out_buf,
                                       size_t buf_len) {
  if (!handle || !handle->service || !key) {
    return 0;
  }
  auto released_data = handle->service->GetReleasedData();
  auto it = released_data.find(key);
  if (it == released_data.end()) {
    return 0;
  }
  const std::string& data = it->second;
  if (buf_len < data.size()) {
    return 0;
  }
  memcpy(out_buf, data.data(), data.size());
  return data.size();
}

size_t fake_data_rw_get_released_data_count(FakeDataReadWriteHandle* handle) {
  if (!handle || !handle->service) {
    return 0;
  }
  return handle->service->GetReleasedData().size();
}

size_t fake_data_rw_get_released_data_key_at(FakeDataReadWriteHandle* handle,
                                              size_t index, char* out_buf,
                                              size_t buf_len) {
  if (!handle || !handle->service) {
    return 0;
  }
  auto released_data = handle->service->GetReleasedData();
  if (index >= released_data.size()) {
    return 0;
  }
  auto it = released_data.begin();
  std::advance(it, index);
  const std::string& key = it->first;
  if (buf_len < key.size() + 1) {
    return 0;
  }
  memcpy(out_buf, key.c_str(), key.size());
  out_buf[key.size()] = '\0';
  return key.size();
}

}  // extern "C"
