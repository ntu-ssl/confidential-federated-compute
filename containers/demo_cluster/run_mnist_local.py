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

"""Standalone MNIST federated training — same logic as PROGRAM_MNIST_TRAINING_TEMPLATE.

Runs the exact same TFF computations, model architecture, and training loop
as the TEE program in main.rs, but loads data directly from binary files
instead of going through the program executor / MinSepDataSource / KMS.

Use this to compare results with the TEE execution.

Usage:
    python run_mnist_local.py \
        --train-file testdata/mnist/train.bin \
        --test-file testdata/mnist/test.bin \
        --num-clients 4 --images-per-client 10000 \
        --num-rounds 4 --local-epochs 5 --learning-rate 0.02
"""

import argparse
import struct
import time

import federated_language
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

# ---------------------------------------------------------------------------
# Constants — identical to PROGRAM_MNIST_TRAINING_TEMPLATE in main.rs
# ---------------------------------------------------------------------------
COMBINED_WIDTH = 785
TOTAL_PARAMS = 101770
RECORD_SIZE = 785  # 1 byte label + 784 bytes pixels

SIZES = [100352, 128, 1280, 10]
SHAPES = [(784, 128), (128,), (128, 10), (10,)]


# ---------------------------------------------------------------------------
# Data loading — replaces MinSepDataSource + encrypted storage
# ---------------------------------------------------------------------------
def load_mnist_binary(path):
    """Load MNIST binary file (from download_mnist.py).

    Each record: [label: 1 byte][pixels: 784 bytes]
    Returns: np.ndarray of shape [N, 785] float32 (pixels/255 + label).
    """
    raw = open(path, "rb").read()
    num_images = len(raw) // RECORD_SIZE
    data = np.zeros((num_images, COMBINED_WIDTH), dtype=np.float32)
    for i in range(num_images):
        off = i * RECORD_SIZE
        label = raw[off]
        pixels = raw[off + 1 : off + 1 + 784]
        data[i, :784] = np.frombuffer(pixels, dtype=np.uint8).astype(np.float32) / 255.0
        data[i, 784] = float(label)
    return data


def split_for_clients(data, num_clients, images_per_client):
    """Split data across clients — same as parse_mnist_for_clients in main.rs."""
    total_needed = num_clients * images_per_client
    assert len(data) >= total_needed, (
        f"Need {total_needed} images ({num_clients} x {images_per_client}), "
        f"but file has {len(data)}"
    )
    clients = []
    for c in range(num_clients):
        start = c * images_per_client
        end = start + images_per_client
        clients.append(data[start:end])
    return clients


# ---------------------------------------------------------------------------
# Model — identical to PROGRAM_MNIST_TRAINING_TEMPLATE
# ---------------------------------------------------------------------------
def _forward(w, client_data):
    """FC forward pass: Flatten(784) -> Dense(128, ReLU) -> Dense(10)."""
    images = client_data[:, :784]
    labels = tf.cast(client_data[:, 784], tf.int32)
    x = tf.nn.relu(tf.matmul(images, w[0]) + w[1])
    logits = tf.matmul(x, w[2]) + w[3]
    return logits, labels


def eval_on_test(w_flat, test_data, label):
    """Evaluate on test set — identical to _eval_on_test in main.rs."""
    w_np = np.array(w_flat)
    w_list = []
    offset = 0
    for sz, sh in zip(SIZES, SHAPES):
        w_list.append(tf.constant(w_np[offset : offset + sz].reshape(sh)))
        offset += sz

    logits, labels = _forward(w_list, tf.constant(test_data))
    preds = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
    labels_np = labels.numpy()
    correct = int(np.sum(preds == labels_np))
    total = len(labels_np)
    loss = float(
        tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits
            )
        ).numpy()
    )
    print(
        f"{label} - test: {correct}/{total} "
        f"({round(100.0 * correct / total, 2)}%), loss: {round(loss, 4)}"
    )
    return correct, total, loss


def glorot_uniform(shape, rng):
    """Glorot uniform — identical to main.rs template."""
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, shape).astype(np.float32)


def init_weights():
    """Glorot uniform for kernels, zeros for biases — identical to main.rs."""
    rng = np.random.RandomState(42)
    parts = []
    for i, (sz, sh) in enumerate(zip(SIZES, SHAPES)):
        if i % 2 == 0:
            parts.append(glorot_uniform(sh, rng).flatten())
        else:
            parts.append(np.zeros(sz, dtype=np.float32))
    return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Standalone MNIST federated training (same as TEE program)"
    )
    parser.add_argument(
        "--train-file",
        default="/home/r14922032/cfc_tensorflow/containers/demo_cluster/testdata/mnist/train.bin",
        help="MNIST training binary (from download_mnist.py)",
    )
    parser.add_argument(
        "--test-file",
        default="/home/r14922032/cfc_tensorflow/containers/demo_cluster/testdata/mnist/test.bin",
        help="MNIST test binary (from download_mnist.py)",
    )
    parser.add_argument("--num-clients", type=int, default=2)
    parser.add_argument("--images-per-client", type=int, default=30000)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=30000//32)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument(
        "--output-weights",
        default="/tmp/mnist_trained_model_local.bin",
        help="Path to save trained weights",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("MNIST Federated Training — Local Reference (no TEE)")
    print("=" * 60)
    print(f"  Train file:        {args.train_file}")
    print(f"  Test file:         {args.test_file}")
    print(f"  Clients:           {args.num_clients}")
    print(f"  Images per client: {args.images_per_client}")
    print(f"  Rounds:            {args.num_rounds}")
    print(f"  Local epochs:      {args.local_epochs}")
    print(f"  Learning rate:     {args.learning_rate}")
    print()

    # --- Set up TFF execution context ---
    try:
        tff.backends.native.set_local_python_execution_context()
    except AttributeError:
        tff.backends.native.set_sync_local_cpp_execution_context()

    # --- Load data ---
    print("Loading training data...")
    train_data = load_mnist_binary(args.train_file)
    print(f"  {len(train_data)} images loaded")

    print("Loading test data...")
    test_data = load_mnist_binary(args.test_file)
    print(f"  {len(test_data)} images loaded")

    client_datasets = split_for_clients(
        train_data, args.num_clients, args.images_per_client
    )
    print(
        f"  Split into {args.num_clients} clients x "
        f"{args.images_per_client} images each"
    )
    print()

    # --- Build TFF computations (identical to main.rs template) ---
    NUM_CLIENTS = args.num_clients
    IMAGES_PER_CLIENT = args.images_per_client
    LOCAL_EPOCHS = args.local_epochs
    LR = args.learning_rate

    weights_type = federated_language.TensorType(np.float32, [TOTAL_PARAMS])
    data_type = federated_language.TensorType(
        np.float32, [IMAGES_PER_CLIENT, COMBINED_WIDTH]
    )
    server_weights_type = federated_language.FederatedType(
        weights_type, federated_language.SERVER
    )
    client_data_type = federated_language.FederatedType(
        data_type, federated_language.CLIENTS
    )

    @tff.tensorflow.computation(weights_type, data_type)
    def client_train(global_weights_flat, client_data):
        w_flat = global_weights_flat
        for _ in range(LOCAL_EPOCHS):
            parts = tf.split(w_flat, SIZES)
            w = [tf.reshape(p, s) for p, s in zip(parts, SHAPES)]
            logits, labels = _forward(w, client_data)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits
                )
            )
            grads = tf.gradients(loss, w)
            updated = [wi - LR * gi for wi, gi in zip(w, grads)]
            w_flat = tf.concat(
                [tf.reshape(u, [-1]) for u in updated], axis=0
            )
        return w_flat

    @federated_language.federated_computation(
        server_weights_type, client_data_type
    )
    def federated_train_round(server_weights, client_data):
        broadcast_weights = federated_language.federated_broadcast(
            server_weights
        )
        client_updated = federated_language.federated_map(
            client_train, (broadcast_weights, client_data)
        )
        return federated_language.federated_mean(client_updated)

    print("TFF computations built successfully.")
    print()

    # --- Initialize weights (identical to main.rs) ---
    weights = init_weights()

    # --- Evaluate baseline ---
    eval_on_test(weights, test_data, f"Round 0/{args.num_rounds}")

    # --- Training loop (identical to main.rs) ---
    for round_num in range(args.num_rounds):
        t0 = time.time()
        result = federated_train_round(weights, client_datasets)
        weights = np.array(result)
        elapsed = time.time() - t0
        total_steps = LOCAL_EPOCHS * NUM_CLIENTS
        ms_per_step = (elapsed * 1000) / total_steps if total_steps > 0 else 0
        eval_on_test(
            weights,
            test_data,
            f"Round {round_num + 1}/{args.num_rounds} "
            f"({elapsed:.1f}s, {ms_per_step:.2f} ms/step)",
        )

    # --- Save weights ---
    weights_bytes = weights.tobytes()
    with open(args.output_weights, "wb") as f:
        f.write(weights_bytes)
    print(f"\nSaved trained weights to {args.output_weights} ({len(weights_bytes)} bytes)")

    print("\nDone!")


if __name__ == "__main__":
    main()
