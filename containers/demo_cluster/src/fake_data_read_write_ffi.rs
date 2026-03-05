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

//! Rust FFI bindings for FakeDataReadWriteService.
//!
//! This module provides a safe Rust wrapper around the C FFI for
//! FakeDataReadWriteService, which is used for testing program executor
//! containers.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::ptr;

/// Opaque handle type for the C++ FakeDataReadWriteService.
#[repr(C)]
struct FakeDataReadWriteHandle {
    _private: [u8; 0],
}

// FFI declarations matching cc/fake_data_read_write_ffi.h
extern "C" {
    fn fake_data_rw_create() -> *mut FakeDataReadWriteHandle;
    fn fake_data_rw_destroy(handle: *mut FakeDataReadWriteHandle);
    fn fake_data_rw_start_server(
        handle: *mut FakeDataReadWriteHandle,
        address: *const c_char,
    ) -> c_int;
    fn fake_data_rw_get_server_address(
        handle: *mut FakeDataReadWriteHandle,
        out_buf: *mut c_char,
        buf_len: usize,
    ) -> usize;
    fn fake_data_rw_store_encrypted(
        handle: *mut FakeDataReadWriteHandle,
        uri: *const c_char,
        message: *const u8,
        message_len: usize,
        blob_id: *const u8,
        blob_id_len: usize,
    ) -> c_int;
    fn fake_data_rw_store_plaintext(
        handle: *mut FakeDataReadWriteHandle,
        uri: *const c_char,
        message: *const u8,
        message_len: usize,
    ) -> c_int;
    fn fake_data_rw_get_input_public_key(
        handle: *mut FakeDataReadWriteHandle,
        out_buf: *mut u8,
        buf_len: usize,
    ) -> usize;
    fn fake_data_rw_get_input_private_key(
        handle: *mut FakeDataReadWriteHandle,
        out_buf: *mut u8,
        buf_len: usize,
    ) -> usize;
    fn fake_data_rw_get_result_public_key(
        handle: *mut FakeDataReadWriteHandle,
        out_buf: *mut u8,
        buf_len: usize,
    ) -> usize;
    fn fake_data_rw_get_result_private_key(
        handle: *mut FakeDataReadWriteHandle,
        out_buf: *mut u8,
        buf_len: usize,
    ) -> usize;
    fn fake_data_rw_get_released_data(
        handle: *mut FakeDataReadWriteHandle,
        key: *const c_char,
        out_buf: *mut u8,
        buf_len: usize,
    ) -> usize;
    fn fake_data_rw_get_released_data_count(handle: *mut FakeDataReadWriteHandle) -> usize;
    fn fake_data_rw_get_released_data_key_at(
        handle: *mut FakeDataReadWriteHandle,
        index: usize,
        out_buf: *mut c_char,
        buf_len: usize,
    ) -> usize;
}

/// Error type for FakeDataReadWriteService operations.
#[derive(Debug, Clone)]
pub struct FakeDataReadWriteError {
    pub message: String,
}

impl std::fmt::Display for FakeDataReadWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for FakeDataReadWriteError {}

/// Result type for FakeDataReadWriteService operations.
pub type Result<T> = std::result::Result<T, FakeDataReadWriteError>;

/// Safe wrapper around the C++ FakeDataReadWriteService.
///
/// This service provides an in-memory implementation of the DataReadWrite
/// gRPC service for testing purposes. It can store encrypted or plaintext
/// messages that can later be retrieved by the program executor TEE.
pub struct FakeDataReadWriteService {
    handle: *mut FakeDataReadWriteHandle,
    server_port: Option<i32>,
}

// Safety: The C++ implementation uses proper synchronization.
unsafe impl Send for FakeDataReadWriteService {}
unsafe impl Sync for FakeDataReadWriteService {}

impl FakeDataReadWriteService {
    /// Creates a new FakeDataReadWriteService instance.
    pub fn new() -> Result<Self> {
        let handle = unsafe { fake_data_rw_create() };
        if handle.is_null() {
            return Err(FakeDataReadWriteError {
                message: "Failed to create FakeDataReadWriteService".to_string(),
            });
        }
        Ok(Self { handle, server_port: None })
    }

    /// Starts the gRPC server on the given address.
    ///
    /// Use "[::1]:0" to let the system choose a random available port.
    /// Returns the actual port number.
    pub fn start_server(&mut self, address: &str) -> Result<i32> {
        let c_address = CString::new(address)
            .map_err(|e| FakeDataReadWriteError { message: format!("Invalid address: {}", e) })?;

        let port = unsafe { fake_data_rw_start_server(self.handle, c_address.as_ptr()) };

        if port < 0 {
            return Err(FakeDataReadWriteError { message: "Failed to start server".to_string() });
        }

        self.server_port = Some(port);
        Ok(port)
    }

    /// Gets the server address after starting.
    pub fn get_server_address(&self) -> Option<String> {
        let mut buf = vec![0u8; 256];
        let len = unsafe {
            fake_data_rw_get_server_address(self.handle, buf.as_mut_ptr() as *mut c_char, buf.len())
        };

        if len == 0 {
            return None;
        }

        buf.truncate(len);
        String::from_utf8(buf).ok()
    }

    /// Stores an encrypted message for the given URI.
    ///
    /// This encrypts the message using HPKE with the input public key,
    /// so it can be decrypted by the TEE using the corresponding private key.
    pub fn store_encrypted_message_for_kms(
        &self,
        uri: &str,
        message: &[u8],
        blob_id: Option<&[u8]>,
    ) -> Result<()> {
        let c_uri = CString::new(uri)
            .map_err(|e| FakeDataReadWriteError { message: format!("Invalid URI: {}", e) })?;

        let (blob_id_ptr, blob_id_len) = match blob_id {
            Some(id) => (id.as_ptr(), id.len()),
            None => (ptr::null(), 0),
        };

        let result = unsafe {
            fake_data_rw_store_encrypted(
                self.handle,
                c_uri.as_ptr(),
                message.as_ptr(),
                message.len(),
                blob_id_ptr,
                blob_id_len,
            )
        };

        if result != 0 {
            return Err(FakeDataReadWriteError {
                message: "Failed to store encrypted message".to_string(),
            });
        }

        Ok(())
    }

    /// Stores a plaintext message for the given URI.
    pub fn store_plaintext_message(&self, uri: &str, message: &[u8]) -> Result<()> {
        let c_uri = CString::new(uri)
            .map_err(|e| FakeDataReadWriteError { message: format!("Invalid URI: {}", e) })?;

        let result = unsafe {
            fake_data_rw_store_plaintext(
                self.handle,
                c_uri.as_ptr(),
                message.as_ptr(),
                message.len(),
            )
        };

        if result != 0 {
            return Err(FakeDataReadWriteError {
                message: "Failed to store plaintext message".to_string(),
            });
        }

        Ok(())
    }

    /// Gets the input public/private key pair.
    ///
    /// Returns (public_key, private_key).
    pub fn get_input_key_pair(&self) -> (Vec<u8>, Vec<u8>) {
        let mut public_key = vec![0u8; 4096];
        let mut private_key = vec![0u8; 4096];

        let pub_len = unsafe {
            fake_data_rw_get_input_public_key(
                self.handle,
                public_key.as_mut_ptr(),
                public_key.len(),
            )
        };

        let priv_len = unsafe {
            fake_data_rw_get_input_private_key(
                self.handle,
                private_key.as_mut_ptr(),
                private_key.len(),
            )
        };

        public_key.truncate(pub_len);
        private_key.truncate(priv_len);

        (public_key, private_key)
    }

    /// Gets the result public/private key pair.
    ///
    /// Returns (public_key, private_key).
    pub fn get_result_key_pair(&self) -> (Vec<u8>, Vec<u8>) {
        let mut public_key = vec![0u8; 4096];
        let mut private_key = vec![0u8; 4096];

        let pub_len = unsafe {
            fake_data_rw_get_result_public_key(
                self.handle,
                public_key.as_mut_ptr(),
                public_key.len(),
            )
        };

        let priv_len = unsafe {
            fake_data_rw_get_result_private_key(
                self.handle,
                private_key.as_mut_ptr(),
                private_key.len(),
            )
        };

        public_key.truncate(pub_len);
        private_key.truncate(priv_len);

        (public_key, private_key)
    }

    /// Gets released data for a specific key.
    pub fn get_released_data(&self, key: &str) -> Option<Vec<u8>> {
        let c_key = CString::new(key).ok()?;
        let mut buf = vec![0u8; 65536]; // 64KB buffer

        let len = unsafe {
            fake_data_rw_get_released_data(self.handle, c_key.as_ptr(), buf.as_mut_ptr(), buf.len())
        };

        if len == 0 {
            return None;
        }

        buf.truncate(len);
        Some(buf)
    }

    /// Gets all released data as a map.
    pub fn get_all_released_data(&self) -> HashMap<String, Vec<u8>> {
        let count = unsafe { fake_data_rw_get_released_data_count(self.handle) };
        let mut result = HashMap::new();

        for i in 0..count {
            let mut key_buf = vec![0u8; 1024];
            let key_len = unsafe {
                fake_data_rw_get_released_data_key_at(
                    self.handle,
                    i,
                    key_buf.as_mut_ptr() as *mut c_char,
                    key_buf.len(),
                )
            };

            if key_len == 0 {
                continue;
            }

            key_buf.truncate(key_len);
            if let Ok(key) = String::from_utf8(key_buf) {
                if let Some(data) = self.get_released_data(&key) {
                    result.insert(key, data);
                }
            }
        }

        result
    }
}

impl Default for FakeDataReadWriteService {
    fn default() -> Self {
        Self::new().expect("Failed to create FakeDataReadWriteService")
    }
}

impl Drop for FakeDataReadWriteService {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                fake_data_rw_destroy(self.handle);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to print bytes as hex
    fn hex_preview(data: &[u8], max_len: usize) -> String {
        let preview: Vec<String> =
            data.iter().take(max_len).map(|b| format!("{:02x}", b)).collect();
        if data.len() > max_len {
            format!("[{}... ({} bytes total)]", preview.join(" "), data.len())
        } else {
            format!("[{}]", preview.join(" "))
        }
    }

    // ===== Basic Lifecycle Tests =====

    #[test]
    fn test_create_and_drop() {
        println!("=== test_create_and_drop ===");
        let service = FakeDataReadWriteService::new().unwrap();
        println!("Created FakeDataReadWriteService successfully");
        drop(service);
        println!("Dropped service successfully");
    }

    #[test]
    fn test_default_trait() {
        println!("=== test_default_trait ===");
        let service = FakeDataReadWriteService::default();
        println!("Created service via Default trait");
        drop(service);
    }

    // ===== Server Tests =====

    #[test]
    fn test_start_server() {
        println!("=== test_start_server ===");
        let mut service = FakeDataReadWriteService::new().unwrap();
        let port = service.start_server("[::1]:0").unwrap();
        println!("Server started on port: {}", port);
        assert!(port > 0);

        let address = service.get_server_address();
        println!("Server address: {:?}", address);
        assert!(address.is_some());
        assert!(address.unwrap().contains(&port.to_string()));
    }

    #[test]
    fn test_get_server_address_before_start() {
        println!("=== test_get_server_address_before_start ===");
        let service = FakeDataReadWriteService::new().unwrap();
        let address = service.get_server_address();
        println!("Address before start: {:?}", address);
        assert!(address.is_none());
    }

    // ===== Key Pair Tests =====

    #[test]
    fn test_get_key_pairs() {
        println!("=== test_get_key_pairs ===");
        let service = FakeDataReadWriteService::new().unwrap();

        let (pub_key, priv_key) = service.get_input_key_pair();
        println!("Input public key:  {} bytes - {}", pub_key.len(), hex_preview(&pub_key, 16));
        println!("Input private key: {} bytes - {}", priv_key.len(), hex_preview(&priv_key, 16));
        assert!(!pub_key.is_empty());
        assert!(!priv_key.is_empty());

        let (result_pub, result_priv) = service.get_result_key_pair();
        println!(
            "Result public key:  {} bytes - {}",
            result_pub.len(),
            hex_preview(&result_pub, 16)
        );
        println!(
            "Result private key: {} bytes - {}",
            result_priv.len(),
            hex_preview(&result_priv, 16)
        );
        assert!(!result_pub.is_empty());
        assert!(!result_priv.is_empty());
    }

    #[test]
    fn test_key_pairs_are_different() {
        println!("=== test_key_pairs_are_different ===");
        let service = FakeDataReadWriteService::new().unwrap();

        let (input_pub, input_priv) = service.get_input_key_pair();
        let (result_pub, result_priv) = service.get_result_key_pair();

        println!("Input pub:  {}", hex_preview(&input_pub, 8));
        println!("Result pub: {}", hex_preview(&result_pub, 8));
        println!("Keys are different: {}", input_pub != result_pub);

        assert_ne!(input_pub, result_pub);
        assert_ne!(input_priv, result_priv);
    }

    #[test]
    fn test_key_pairs_consistent() {
        println!("=== test_key_pairs_consistent ===");
        let service = FakeDataReadWriteService::new().unwrap();

        let (pub1, priv1) = service.get_input_key_pair();
        let (pub2, priv2) = service.get_input_key_pair();

        println!("First call:  {}", hex_preview(&pub1, 8));
        println!("Second call: {}", hex_preview(&pub2, 8));
        println!("Keys match: {}", pub1 == pub2);

        assert_eq!(pub1, pub2);
        assert_eq!(priv1, priv2);
    }

    // ===== Store Plaintext Message Tests =====

    #[test]
    fn test_store_plaintext_message() {
        println!("=== test_store_plaintext_message ===");
        let service = FakeDataReadWriteService::new().unwrap();
        let uri = "test/uri";
        let message = b"Hello, World!";
        println!("Storing plaintext at URI: {}", uri);
        println!("Message: {:?} ({} bytes)", String::from_utf8_lossy(message), message.len());
        service.store_plaintext_message(uri, message).unwrap();
        println!("Store successful!");
    }

    #[test]
    fn test_store_multiple_plaintext_messages() {
        println!("=== test_store_multiple_plaintext_messages ===");
        let service = FakeDataReadWriteService::new().unwrap();

        let messages = [("uri_1", "message_1"), ("uri_2", "message_2"), ("uri_3", "message_3")];

        for (uri, msg) in &messages {
            println!("Storing at {}: {:?}", uri, msg);
            service.store_plaintext_message(uri, msg.as_bytes()).unwrap();
        }
        println!("All {} messages stored successfully!", messages.len());
    }

    #[test]
    fn test_store_plaintext_empty_message() {
        println!("=== test_store_plaintext_empty_message ===");
        let service = FakeDataReadWriteService::new().unwrap();
        println!("Storing empty message at 'empty/uri'");
        service.store_plaintext_message("empty/uri", b"").unwrap();
        println!("Empty message stored successfully!");
    }

    #[test]
    fn test_store_plaintext_binary_data() {
        println!("=== test_store_plaintext_binary_data ===");
        let service = FakeDataReadWriteService::new().unwrap();
        let binary_data: Vec<u8> = vec![0x00, 0x01, 0x02, 0xFF, 0xFE, 0x00, 0x42];
        println!("Storing binary data: {}", hex_preview(&binary_data, 16));
        service.store_plaintext_message("binary/uri", &binary_data).unwrap();
        println!("Binary data stored successfully!");
    }

    // ===== Store Encrypted Message Tests =====

    #[test]
    fn test_store_encrypted_message() {
        println!("=== test_store_encrypted_message ===");
        let service = FakeDataReadWriteService::new().unwrap();
        let uri = "encrypted/uri";
        let message = b"secret data";
        println!("Storing encrypted message at URI: {}", uri);
        println!("Plaintext: {:?} ({} bytes)", String::from_utf8_lossy(message), message.len());
        service.store_encrypted_message_for_kms(uri, message, None).unwrap();
        println!("Encrypted message stored successfully!");
    }

    #[test]
    fn test_store_encrypted_message_with_blob_id() {
        println!("=== test_store_encrypted_message_with_blob_id ===");
        let service = FakeDataReadWriteService::new().unwrap();
        let blob_id = b"my_blob_id_12345";
        println!("Storing with blob_id: {:?}", String::from_utf8_lossy(blob_id));
        service
            .store_encrypted_message_for_kms("encrypted/uri", b"secret data", Some(blob_id))
            .unwrap();
        println!("Encrypted message with blob_id stored successfully!");
    }

    #[test]
    fn test_store_multiple_encrypted_messages() {
        println!("=== test_store_multiple_encrypted_messages ===");
        let service = FakeDataReadWriteService::new().unwrap();

        println!("Storing encrypted message 1...");
        service.store_encrypted_message_for_kms("enc/uri_1", b"secret_1", None).unwrap();
        println!("Storing encrypted message 2...");
        service.store_encrypted_message_for_kms("enc/uri_2", b"secret_2", None).unwrap();
        println!("Both encrypted messages stored successfully!");
    }

    // ===== Released Data Tests =====

    #[test]
    fn test_get_released_data_empty() {
        println!("=== test_get_released_data_empty ===");
        let service = FakeDataReadWriteService::new().unwrap();

        let data = service.get_released_data("nonexistent");
        println!("Released data for 'nonexistent': {:?}", data);
        assert!(data.is_none());

        let all_data = service.get_all_released_data();
        println!("All released data count: {}", all_data.len());
        assert!(all_data.is_empty());
    }

    #[test]
    fn test_get_all_released_data_empty() {
        println!("=== test_get_all_released_data_empty ===");
        let service = FakeDataReadWriteService::new().unwrap();
        let released = service.get_all_released_data();
        println!("Released data map: {:?}", released);
        assert!(released.is_empty());
    }

    // ===== Error Handling Tests =====

    #[test]
    fn test_store_plaintext_duplicate_uri_fails() {
        println!("=== test_store_plaintext_duplicate_uri_fails ===");
        let service = FakeDataReadWriteService::new().unwrap();

        println!("First store at 'duplicate/uri'...");
        service.store_plaintext_message("duplicate/uri", b"first").unwrap();
        println!("First store succeeded!");

        println!("Second store at same URI...");
        let result = service.store_plaintext_message("duplicate/uri", b"second");
        println!("Second store result: {:?}", result);
        assert!(result.is_err());
        println!("Correctly rejected duplicate URI!");
    }

    #[test]
    fn test_store_encrypted_duplicate_uri_fails() {
        println!("=== test_store_encrypted_duplicate_uri_fails ===");
        let service = FakeDataReadWriteService::new().unwrap();

        println!("First encrypted store...");
        service.store_encrypted_message_for_kms("duplicate/enc", b"first", None).unwrap();
        println!("First store succeeded!");

        println!("Second encrypted store at same URI...");
        let result = service.store_encrypted_message_for_kms("duplicate/enc", b"second", None);
        println!("Second store result: {:?}", result);
        assert!(result.is_err());
        println!("Correctly rejected duplicate URI!");
    }

    #[test]
    fn test_mixed_duplicate_uri_fails() {
        println!("=== test_mixed_duplicate_uri_fails ===");
        let service = FakeDataReadWriteService::new().unwrap();

        println!("First: store plaintext...");
        service.store_plaintext_message("mixed/uri", b"plaintext").unwrap();
        println!("Plaintext store succeeded!");

        println!("Second: try encrypted at same URI...");
        let result = service.store_encrypted_message_for_kms("mixed/uri", b"encrypted", None);
        println!("Encrypted store result: {:?}", result);
        assert!(result.is_err());
        println!("Correctly rejected mixed duplicate URI!");
    }

    // ===== Integration-style Tests =====

    #[test]
    fn test_full_workflow() {
        println!("=== test_full_workflow ===");
        println!("\n--- Step 1: Create service and start server ---");
        let mut service = FakeDataReadWriteService::new().unwrap();
        let port = service.start_server("[::1]:0").unwrap();
        println!("Server started on port: {}", port);
        assert!(port > 0);

        println!("\n--- Step 2: Get key pairs ---");
        let (input_pub, input_priv) = service.get_input_key_pair();
        let (result_pub, result_priv) = service.get_result_key_pair();
        println!("Input public key:  {} bytes - {}", input_pub.len(), hex_preview(&input_pub, 16));
        println!(
            "Input private key: {} bytes - {}",
            input_priv.len(),
            hex_preview(&input_priv, 16)
        );
        println!(
            "Result public key:  {} bytes - {}",
            result_pub.len(),
            hex_preview(&result_pub, 16)
        );
        println!(
            "Result private key: {} bytes - {}",
            result_priv.len(),
            hex_preview(&result_priv, 16)
        );
        assert!(!input_pub.is_empty());
        assert!(!result_pub.is_empty());

        println!("\n--- Step 3: Store test data ---");
        println!("Storing plaintext for client1...");
        service.store_plaintext_message("client1/data", b"tensor_data_1").unwrap();
        println!("Storing plaintext for client2...");
        service.store_plaintext_message("client2/data", b"tensor_data_2").unwrap();
        println!("Storing encrypted for client3...");
        service.store_encrypted_message_for_kms("client3/data", b"encrypted_tensor", None).unwrap();
        println!("All test data stored!");

        println!("\n--- Step 4: Verify server address ---");
        let address = service.get_server_address().unwrap();
        println!("Server address: {}", address);
        assert!(address.contains(&port.to_string()));

        println!("\n--- Step 5: Check released data (should be empty) ---");
        let released = service.get_all_released_data();
        println!("Released data count: {}", released.len());
        for (key, value) in &released {
            println!("  {}: {} bytes", key, value.len());
        }

        println!("\n=== Full workflow completed successfully! ===");
    }
}
