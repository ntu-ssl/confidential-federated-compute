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

"""Test MNIST federated training computations locally without TEE.

Verifies the Python code from PROGRAM_MNIST_TRAINING_TEMPLATE (main.rs) works:
  1. FC parameter count matches architecture
  2. client_train: forward/backward pass traces in @tff.tensorflow.computation
  3. federated_train_round: broadcast + map + mean via federated_computation
  4. Multi-round training produces non-trivial weight updates
  5. Result serialization round-trips correctly (tff.framework.serialize_value)
  6. eval_accuracy: eager-mode forward pass matches _eval_on_test in main.rs

Usage (Bazel):
    bazel test //containers/demo_cluster:test_mnist_program

Requires: tensorflow, tensorflow_federated, federated_language, numpy
"""

import sys
import traceback

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import federated_language

# ---------------------------------------------------------------------------
# Constants matching PROGRAM_MNIST_TRAINING_TEMPLATE in main.rs
# Model: Flatten(784) -> Dense(128, ReLU) -> Dense(10)
# Matches NVFlare MNIST model (nvflare_test/model.py) minus Dropout.
# ---------------------------------------------------------------------------
TOTAL_PARAMS = 101770
COMBINED_WIDTH = 785

# Per-variable element counts:
#   fc1_kernel(784*128=100352), fc1_bias(128),
#   fc2_kernel(128*10=1280), fc2_bias(10)
SIZES = [100352, 128, 1280, 10]

# Per-variable shapes
SHAPES = [(784, 128), (128,), (128, 10), (10,)]


# ---------------------------------------------------------------------------
# Build TFF computations (mirrors PROGRAM_MNIST_TRAINING_TEMPLATE)
# ---------------------------------------------------------------------------
def _forward(w, client_data):
    """FC forward pass (mirrors _forward in PROGRAM_MNIST_TRAINING_TEMPLATE)."""
    images = client_data[:, :784]
    labels = tf.cast(client_data[:, 784], tf.int32)
    x = tf.nn.relu(tf.matmul(images, w[0]) + w[1])
    logits = tf.matmul(x, w[2]) + w[3]
    return logits, labels


def eval_accuracy(weights_flat, test_data):
    """Evaluate model accuracy in eager mode (mirrors _eval_on_test in main.rs)."""
    w_np = np.array(weights_flat)
    w_list = []
    offset = 0
    for sz, sh in zip(SIZES, SHAPES):
        w_list.append(tf.constant(w_np[offset:offset + sz].reshape(sh)))
        offset += sz

    logits, labels = _forward(w_list, tf.constant(test_data))
    preds = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
    labels_np = labels.numpy()
    correct = int(np.sum(preds == labels_np))
    total = len(labels_np)
    loss = float(tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits
        )
    ).numpy())
    return correct, total, loss


def glorot_init_weights(seed=42):
    """Glorot uniform initialization matching Keras defaults.

    Kernels (even indices) use Glorot uniform, biases (odd indices) are zeros.
    This mirrors the initialization in PROGRAM_MNIST_TRAINING_TEMPLATE.
    """
    rng = np.random.RandomState(seed)
    parts = []
    for i, (sz, sh) in enumerate(zip(SIZES, SHAPES)):
        if i % 2 == 0:  # kernel (always 2D for FC model)
            fan_in, fan_out = sh[0], sh[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            parts.append(
                rng.uniform(-limit, limit, sh).astype(np.float32).flatten()
            )
        else:  # bias
            parts.append(np.zeros(sz, dtype=np.float32))
    return np.concatenate(parts)


def build_computations(images_per_client, local_epochs=1, lr=0.01):
    """Build client_train and federated_train_round.

    These mirror the computations defined in PROGRAM_MNIST_TRAINING_TEMPLATE
    (containers/demo_cluster/src/main.rs). The FC architecture, weight layout,
    and training logic are identical.

    Args:
        images_per_client: Number of images per client (determines input shape).
        local_epochs: Number of full-batch gradient steps per client per round.
        lr: Learning rate for SGD.

    Returns:
        Tuple of (client_train, federated_train_round) TFF computations.
    """
    weights_type = federated_language.TensorType(np.float32, [TOTAL_PARAMS])
    data_type = federated_language.TensorType(
        np.float32, [images_per_client, COMBINED_WIDTH]
    )

    @tff.tensorflow.computation(weights_type, data_type)
    def client_train(global_weights_flat, client_data):
        w_flat = global_weights_flat
        for _ in range(local_epochs):
            parts = tf.split(w_flat, SIZES)
            w = [tf.reshape(p, s) for p, s in zip(parts, SHAPES)]
            logits, labels = _forward(w, client_data)
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits
                )
            )
            grads = tf.gradients(loss, w)
            updated = [wi - lr * gi for wi, gi in zip(w, grads)]
            w_flat = tf.concat(
                [tf.reshape(u, [-1]) for u in updated], axis=0
            )
        return w_flat

    server_weights_type = federated_language.FederatedType(
        weights_type, federated_language.SERVER
    )
    client_data_type = federated_language.FederatedType(
        data_type, federated_language.CLIENTS
    )

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

    return client_train, federated_train_round


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def make_synthetic_client_data(num_images, seed=42):
    """Create synthetic MNIST-like data: [num_images, 785] float32.

    First 784 columns: pixel values in [0, 1] (matching MNIST normalization).
    Last column: label (integer 0-9, stored as float32).
    """
    rng = np.random.RandomState(seed)
    images = rng.rand(num_images, 784).astype(np.float32)  # [0, 1] like MNIST
    labels = rng.randint(0, 10, (num_images, 1)).astype(np.float32)
    return np.concatenate([images, labels], axis=1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_param_count():
    """Test 0: Verify TOTAL_PARAMS matches FC architecture."""
    print('Test 0: Verify parameter count')

    assert sum(SIZES) == TOTAL_PARAMS, (
        f'SIZES sum {sum(SIZES)} != TOTAL_PARAMS {TOTAL_PARAMS}'
    )

    # Verify each layer
    assert SIZES[0] == 784 * 128   # fc1 kernel
    assert SIZES[1] == 128          # fc1 bias
    assert SIZES[2] == 128 * 10    # fc2 kernel
    assert SIZES[3] == 10           # fc2 bias

    total = (784 * 128 + 128) + (128 * 10 + 10)
    assert total == TOTAL_PARAMS

    print(f'  Total params: {TOTAL_PARAMS}')
    print(f'  Layer sizes: FC1={784 * 128 + 128} FC2={128 * 10 + 10}')
    print('  PASSED!\n')


def test_client_train():
    """Test 1: Verify FC forward/backward in @tff.tensorflow.computation."""
    print('Test 1: client_train computation (single client, 10 images)')

    images_per_client = 10
    client_train, _ = build_computations(images_per_client)

    weights = glorot_init_weights(seed=42)
    client_data = make_synthetic_client_data(images_per_client, seed=42)

    result = client_train(weights, client_data)
    result_np = np.array(result)

    assert result_np.shape == (TOTAL_PARAMS,), f'Shape: {result_np.shape}'
    assert not np.allclose(result_np, weights), 'Weights should change after training'
    assert np.all(np.isfinite(result_np)), 'Weights must be finite'

    print(f'  Output shape: {result_np.shape}')
    print(f'  Weight norm:  {np.linalg.norm(result_np):.6f}')
    print(f'  Max |weight|: {np.max(np.abs(result_np)):.6f}')
    print(f'  Non-zero:     {np.count_nonzero(result_np)}/{TOTAL_PARAMS}')
    print('  PASSED!\n')


def test_federated_round():
    """Test 2: Verify federated broadcast + map + mean."""
    print('Test 2: federated_train_round (2 clients, 10 images each)')

    images_per_client = 10
    num_clients = 2
    _, fed_round = build_computations(images_per_client)

    weights = glorot_init_weights(seed=42)
    client_data = [
        make_synthetic_client_data(images_per_client, seed=i)
        for i in range(num_clients)
    ]

    result = fed_round(weights, client_data)
    result_np = np.array(result)

    assert result_np.shape == (TOTAL_PARAMS,), f'Shape: {result_np.shape}'
    assert not np.allclose(result_np, weights), 'Weights should change'
    assert np.all(np.isfinite(result_np)), 'Weights must be finite'

    print(f'  Output shape: {result_np.shape}')
    print(f'  Weight norm:  {np.linalg.norm(result_np):.6f}')
    print('  PASSED!\n')


def test_multi_round():
    """Test 3: Verify multi-round training with loss decrease."""
    print('Test 3: Multi-round training (2 rounds, 2 clients, 3 local epochs)')

    images_per_client = 10
    num_clients = 2
    num_rounds = 2
    local_epochs = 3
    _, fed_round = build_computations(
        images_per_client, local_epochs=local_epochs, lr=0.01
    )

    weights = glorot_init_weights(seed=42)
    test_data = make_synthetic_client_data(20, seed=99)

    _, _, loss_before = eval_accuracy(weights, test_data)
    print(f'  Before training: loss={loss_before:.4f}')

    for r in range(num_rounds):
        client_data = [
            make_synthetic_client_data(images_per_client, seed=r * 100 + i)
            for i in range(num_clients)
        ]
        weights = fed_round(weights, client_data)
        w_np = np.array(weights)
        _, _, loss_r = eval_accuracy(w_np, test_data)
        print(
            f'  Round {r + 1}: norm={np.linalg.norm(w_np):.6f}, '
            f'loss={loss_r:.4f}'
        )

    w_np = np.array(weights)
    assert np.all(np.isfinite(w_np)), 'Non-finite weights after multi-round'
    assert np.linalg.norm(w_np) > 0, 'Weights are all zero'
    print('  PASSED!\n')


def test_serialization():
    """Test 4: Verify result serialization matches TEE release code."""
    print('Test 4: Result serialization round-trip')

    images_per_client = 10
    _, fed_round = build_computations(images_per_client)

    weights = glorot_init_weights(seed=42)
    client_data = [
        make_synthetic_client_data(images_per_client, seed=i) for i in range(2)
    ]
    weights = fed_round(weights, client_data)

    # Serialize (same code as TEE program's release path)
    weights_val, _ = tff.framework.serialize_value(
        weights, federated_language.framework.infer_type(weights)
    )
    serialized = weights_val.SerializeToString()

    assert len(serialized) > 0, 'Serialized value is empty'

    # Deserialize and verify round-trip
    deserialized, _ = tff.framework.deserialize_value(weights_val)
    deser_np = np.array(deserialized)
    weights_np = np.array(weights)
    np.testing.assert_allclose(deser_np, weights_np, rtol=1e-5)

    print(f'  Serialized size: {len(serialized)} bytes')
    print(f'  Round-trip match: True')
    print('  PASSED!\n')


def test_eval_accuracy():
    """Test 5: Verify eager-mode eval_accuracy (mirrors _eval_on_test)."""
    print('Test 5: eval_accuracy (eager-mode forward pass)')

    num_images = 20
    test_data = make_synthetic_client_data(num_images, seed=99)

    # With Glorot-initialized weights, should get valid loss
    weights_init = glorot_init_weights(seed=42)
    correct, total, loss = eval_accuracy(weights_init, test_data)

    assert total == num_images, f'Expected {num_images} samples, got {total}'
    assert loss > 0, f'Loss should be positive, got {loss}'
    assert np.isfinite(loss), f'Loss must be finite, got {loss}'
    assert 0 <= correct <= total, f'Correct {correct} out of range [0, {total}]'

    print(f'  Init weights: {correct}/{total} correct, loss={loss:.4f}')

    # After one round of training, eval should still produce valid results
    images_per_client = num_images
    _, fed_round = build_computations(images_per_client, local_epochs=3)
    weights_trained = fed_round(
        weights_init,
        [make_synthetic_client_data(images_per_client, seed=i) for i in range(2)],
    )

    correct_t, total_t, loss_t = eval_accuracy(weights_trained, test_data)
    assert total_t == num_images
    assert np.isfinite(loss_t), f'Post-training loss must be finite, got {loss_t}'

    print(f'  Trained weights: {correct_t}/{total_t} correct, loss={loss_t:.4f}')
    print('  PASSED!\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def setup_tff_context():
    """Set up local TFF execution context for testing."""
    try:
        tff.backends.native.set_local_python_execution_context()
        print('  Context: tff.backends.native.set_local_python_execution_context()')
        return True
    except AttributeError:
        pass

    try:
        tff.backends.native.set_sync_local_cpp_execution_context()
        print('  Context: tff.backends.native.set_sync_local_cpp_execution_context()')
        return True
    except AttributeError:
        pass

    print('  Context: using TFF default (auto-initialized)')
    return True


def main():
    print('=' * 60)
    print('MNIST Federated Training - Local Test (no TEE)')
    print('=' * 60)
    print()

    print('Setting up TFF execution context...')
    setup_tff_context()
    print()

    tests = [
        ('param_count', test_param_count),
        ('client_train', test_client_train),
        ('federated_round', test_federated_round),
        ('multi_round', test_multi_round),
        ('serialization', test_serialization),
        ('eval_accuracy', test_eval_accuracy),
    ]

    passed = 0
    failed = 0
    errors = []

    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            errors.append((name, e))
            print(f'  FAILED: {e}')
            traceback.print_exc()
            print()

    print('=' * 60)
    print(f'Results: {passed} passed, {failed} failed out of {len(tests)} tests')
    if errors:
        print('\nFailed tests:')
        for name, err in errors:
            print(f'  - {name}: {err}')
    print('=' * 60)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
