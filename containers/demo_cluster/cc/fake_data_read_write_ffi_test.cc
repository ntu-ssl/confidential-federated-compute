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

#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace {

TEST(FakeDataReadWriteFfiTest, CreateAndDestroy) {
  FakeDataReadWriteHandle* handle = fake_data_rw_create();
  ASSERT_NE(handle, nullptr);
  fake_data_rw_destroy(handle);
}

TEST(FakeDataReadWriteFfiTest, StartServer) {
  FakeDataReadWriteHandle* handle = fake_data_rw_create();
  ASSERT_NE(handle, nullptr);

  int port = fake_data_rw_start_server(handle, "[::1]:0");
  EXPECT_GT(port, 0);

  char address[256];
  size_t len = fake_data_rw_get_server_address(handle, address, sizeof(address));
  EXPECT_GT(len, 0);
  EXPECT_TRUE(strstr(address, std::to_string(port).c_str()) != nullptr);

  fake_data_rw_destroy(handle);
}

TEST(FakeDataReadWriteFfiTest, GetKeyPairs) {
  FakeDataReadWriteHandle* handle = fake_data_rw_create();
  ASSERT_NE(handle, nullptr);

  std::vector<uint8_t> public_key(1024);
  std::vector<uint8_t> private_key(1024);

  // Test input key pair
  size_t pub_len =
      fake_data_rw_get_input_public_key(handle, public_key.data(), public_key.size());
  EXPECT_GT(pub_len, 0);

  size_t priv_len =
      fake_data_rw_get_input_private_key(handle, private_key.data(), private_key.size());
  EXPECT_GT(priv_len, 0);

  // Test result key pair
  pub_len =
      fake_data_rw_get_result_public_key(handle, public_key.data(), public_key.size());
  EXPECT_GT(pub_len, 0);

  priv_len =
      fake_data_rw_get_result_private_key(handle, private_key.data(), private_key.size());
  EXPECT_GT(priv_len, 0);

  fake_data_rw_destroy(handle);
}

TEST(FakeDataReadWriteFfiTest, StorePlaintext) {
  FakeDataReadWriteHandle* handle = fake_data_rw_create();
  ASSERT_NE(handle, nullptr);

  const char* uri = "test/uri";
  const std::string message = "Hello, World!";

  int result = fake_data_rw_store_plaintext(
      handle, uri, reinterpret_cast<const uint8_t*>(message.data()),
      message.size());
  EXPECT_EQ(result, 0);

  fake_data_rw_destroy(handle);
}

TEST(FakeDataReadWriteFfiTest, NullHandleReturnsError) {
  EXPECT_EQ(fake_data_rw_start_server(nullptr, "[::1]:0"), -1);
  EXPECT_EQ(fake_data_rw_store_plaintext(nullptr, "uri", nullptr, 0), -1);
  EXPECT_EQ(fake_data_rw_get_input_public_key(nullptr, nullptr, 0), 0);
}

}  // namespace
