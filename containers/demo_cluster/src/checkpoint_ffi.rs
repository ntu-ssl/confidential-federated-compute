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

//! Rust FFI bindings for FederatedCompute checkpoint generation.
//!
//! This module provides a safe Rust wrapper around the TensorFlow Federated
//! C++ checkpoint generation library via FFI.

use std::ffi::{c_char, c_float, c_int, CString};

// FFI declarations matching cc/checkpoint_ffi.h
extern "C" {
    fn checkpoint_build_from_ints(
        values: *const c_int,
        values_len: usize,
        tensor_name: *const c_char,
        out_buf: *mut u8,
        buf_len: usize,
    ) -> usize;

    fn checkpoint_build_from_floats(
        values: *const c_float,
        values_len: usize,
        shape: *const i32,
        shape_len: usize,
        tensor_name: *const c_char,
        out_buf: *mut u8,
        buf_len: usize,
    ) -> usize;

    fn checkpoint_build_float_int_tensors(
        float_data: *const c_float,
        float_data_len: usize,
        float_shape: *const i32,
        float_shape_len: usize,
        float_tensor_name: *const c_char,
        int_data: *const i32,
        int_data_len: usize,
        int_shape: *const i32,
        int_shape_len: usize,
        int_tensor_name: *const c_char,
        out_buf: *mut u8,
        buf_len: usize,
    ) -> usize;
}

/// Error type for checkpoint operations.
#[derive(Debug, Clone)]
pub struct CheckpointError {
    pub message: String,
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CheckpointError {}

/// Result type for checkpoint operations.
pub type Result<T> = std::result::Result<T, CheckpointError>;

/// Builds a FederatedCompute checkpoint from integer values.
///
/// Creates a checkpoint in FCv1 format with a single INT32 tensor.
/// Uses the TensorFlow Federated C++ library via FFI.
///
/// # Arguments
/// * `values` - The integer values to include in the tensor
/// * `tensor_name` - The name of the tensor in the checkpoint
///
/// # Returns
/// The checkpoint data as bytes in FCv1 format.
///
/// # Example
/// ```ignore
/// let checkpoint = build_checkpoint_from_ints(&[1, 2, 3], "my_tensor")?;
/// ```
pub fn build_checkpoint_from_ints(values: &[i32], tensor_name: &str) -> Result<Vec<u8>> {
    let c_tensor_name = CString::new(tensor_name).map_err(|e| CheckpointError {
        message: format!("Invalid tensor name: {}", e),
    })?;

    // 64KB buffer should be enough for most checkpoints
    let mut buf = vec![0u8; 65536];

    let len = unsafe {
        checkpoint_build_from_ints(
            values.as_ptr() as *const c_int,
            values.len(),
            c_tensor_name.as_ptr(),
            buf.as_mut_ptr(),
            buf.len(),
        )
    };

    if len == 0 {
        return Err(CheckpointError {
            message: "Failed to build checkpoint (maybe buffer too small or invalid input?)"
                .to_string(),
        });
    }

    buf.truncate(len);
    Ok(buf)
}

/// Builds a FederatedCompute checkpoint from float values with arbitrary shape.
///
/// Creates a checkpoint in FCv1 format with a single FLOAT tensor.
///
/// # Arguments
/// * `values` - The float values
/// * `shape` - The tensor shape (e.g., [N, 3073] for combined CIFAR-10 data)
/// * `tensor_name` - The name of the tensor in the checkpoint
///
/// # Returns
/// The checkpoint data as bytes in FCv1 format.
pub fn build_checkpoint_from_floats(
    values: &[f32],
    shape: &[i32],
    tensor_name: &str,
) -> Result<Vec<u8>> {
    let c_tensor_name = CString::new(tensor_name).map_err(|e| CheckpointError {
        message: format!("Invalid tensor name: {}", e),
    })?;

    // Dynamic buffer sizing: data size + overhead for FCv1 format
    let estimated_size = values.len() * 4 + 65536;
    let mut buf = vec![0u8; estimated_size];

    let len = unsafe {
        checkpoint_build_from_floats(
            values.as_ptr() as *const c_float,
            values.len(),
            shape.as_ptr(),
            shape.len(),
            c_tensor_name.as_ptr(),
            buf.as_mut_ptr(),
            buf.len(),
        )
    };

    if len == 0 {
        return Err(CheckpointError {
            message: "Failed to build float checkpoint (buffer too small or invalid input?)"
                .to_string(),
        });
    }

    buf.truncate(len);
    Ok(buf)
}

/// Builds a FederatedCompute checkpoint with two tensors: a float32 tensor
/// and an int32 tensor.
///
/// Used for CIFAR-10 data where images (float32) and labels (int32) are
/// stored together in a single checkpoint.
///
/// # Arguments
/// * `float_data` - The float values (e.g., normalized image pixels)
/// * `float_shape` - The shape of the float tensor (e.g., [N, 32, 32, 3])
/// * `float_tensor_name` - The name of the float tensor (e.g., "images")
/// * `int_data` - The int values (e.g., labels)
/// * `int_shape` - The shape of the int tensor (e.g., [N])
/// * `int_tensor_name` - The name of the int tensor (e.g., "labels")
///
/// # Returns
/// The checkpoint data as bytes in FCv1 format.
pub fn build_checkpoint_float_int_tensors(
    float_data: &[f32],
    float_shape: &[i32],
    float_tensor_name: &str,
    int_data: &[i32],
    int_shape: &[i32],
    int_tensor_name: &str,
) -> Result<Vec<u8>> {
    let c_float_name = CString::new(float_tensor_name).map_err(|e| CheckpointError {
        message: format!("Invalid float tensor name: {}", e),
    })?;
    let c_int_name = CString::new(int_tensor_name).map_err(|e| CheckpointError {
        message: format!("Invalid int tensor name: {}", e),
    })?;

    // Dynamic buffer sizing: data size + overhead for FCv1 format
    let estimated_size = float_data.len() * 4 + int_data.len() * 4 + 65536;
    let mut buf = vec![0u8; estimated_size];

    let len = unsafe {
        checkpoint_build_float_int_tensors(
            float_data.as_ptr() as *const c_float,
            float_data.len(),
            float_shape.as_ptr(),
            float_shape.len(),
            c_float_name.as_ptr(),
            int_data.as_ptr() as *const i32,
            int_data.len(),
            int_shape.as_ptr(),
            int_shape.len(),
            c_int_name.as_ptr(),
            buf.as_mut_ptr(),
            buf.len(),
        )
    };

    if len == 0 {
        return Err(CheckpointError {
            message: "Failed to build float+int checkpoint (buffer too small or invalid input?)"
                .to_string(),
        });
    }

    buf.truncate(len);
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_checkpoint_simple() {
        let values = [1, 2, 3, 4, 5];
        let result = build_checkpoint_from_ints(&values, "test_tensor");
        assert!(result.is_ok());

        let checkpoint = result.unwrap();
        // Check FCv1 header
        assert!(checkpoint.len() >= 4);
        assert_eq!(&checkpoint[0..4], b"FCv1");
        println!("Checkpoint size: {} bytes", checkpoint.len());
    }

    #[test]
    fn test_build_checkpoint_empty_values() {
        let values: [i32; 0] = [];
        let result = build_checkpoint_from_ints(&values, "empty_tensor");
        // Empty tensor should still work
        assert!(result.is_ok());
    }

    #[test]
    fn test_build_checkpoint_large_values() {
        let values: Vec<i32> = (0..1000).collect();
        let result = build_checkpoint_from_ints(&values, "large_tensor");
        assert!(result.is_ok());

        let checkpoint = result.unwrap();
        println!("Large checkpoint size: {} bytes", checkpoint.len());
    }

    #[test]
    fn test_build_checkpoint_float_int_tensors() {
        // Simulate 2 CIFAR-10 images (32x32x3) + 2 labels
        let num_images = 2;
        let image_size = 32 * 32 * 3;
        let float_data: Vec<f32> = (0..num_images * image_size)
            .map(|i| (i as f32) / 255.0)
            .collect();
        let int_data: Vec<i32> = vec![3, 7]; // labels

        let result = build_checkpoint_float_int_tensors(
            &float_data,
            &[num_images as i32, 32, 32, 3],
            "images",
            &int_data,
            &[num_images as i32],
            "labels",
        );
        assert!(result.is_ok());

        let checkpoint = result.unwrap();
        assert!(checkpoint.len() >= 4);
        assert_eq!(&checkpoint[0..4], b"FCv1");
        println!(
            "Float+Int checkpoint size: {} bytes (for {} images)",
            checkpoint.len(),
            num_images
        );
    }
}
