#!/usr/bin/env python3
# Copyright 2025 Google LLC.
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

"""Download MNIST and save as binary files for demo_cluster.

Each record is 785 bytes: [label: 1 byte] [pixels: 784 bytes (28x28)]
This matches the CIFAR-10 binary format pattern (label + raw pixels).

Usage:
    python download_mnist.py [output_dir]
    # Default output_dir: testdata/mnist
"""

import os
import sys

import tensorflow as tf


def main():
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "testdata/mnist"
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading MNIST via tf.keras.datasets.mnist...")
    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.mnist.load_data()
    )
    # train_images: (60000, 28, 28) uint8, train_labels: (60000,) uint8
    # test_images:  (10000, 28, 28) uint8, test_labels:  (10000,) uint8

    for name, images, labels in [
        ("train.bin", train_images, train_labels),
        ("test.bin", test_images, test_labels),
    ]:
        path = os.path.join(output_dir, name)
        with open(path, "wb") as f:
            for img, lbl in zip(images, labels):
                f.write(bytes([int(lbl)]))  # 1 byte label
                f.write(img.tobytes())  # 784 bytes pixels (row-major)

        expected_size = len(images) * 785
        actual_size = os.path.getsize(path)
        assert actual_size == expected_size, (
            f"{name}: expected {expected_size} bytes, got {actual_size}"
        )
        print(f"Wrote {len(images)} records to {path} ({actual_size} bytes)")

    print("Done!")


if __name__ == "__main__":
    main()
