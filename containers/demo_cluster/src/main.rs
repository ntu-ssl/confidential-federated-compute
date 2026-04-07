use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use prost::Message;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::{info, info_span, Instrument};

mod attestation_factory;
mod client_simulator;
mod computation_delegation_proxy;
mod key_derivation;
mod kms_client;
mod launcher_module;
mod program_executor_client;

use attestation_factory::create_reference_values_for_extracted_evidence;
use client_simulator::{compute_policy_hash, create_program_executor_access_policy};
use data_read_write_service::RealDataReadWriteService;
use kms_client::{KmsClient, ProstProtoConversionExt};
use launcher_module::{Args as LauncherArgs, Launcher};
use oak_attestation_verification::extract_evidence;
use program_executor_client::ProgramExecutorClient;
use reference_value_proto::oak::attestation::v1::ReferenceValues;

use confidential_transform_proto::fcp::confidentialcompute::blob_metadata::EncryptionMetadata;
use data_read_write_proto::fcp::confidentialcompute::outgoing::WriteRequest;

/// Test type to run
#[derive(Clone, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum TestType {
    /// Test 1: ProgramWithDataSource - Federated aggregation with client data
    #[default]
    #[value(name = "data-source")]
    DataSource,
    /// Test 2: ProgramWithModelLoading - Model loading from zip file
    #[value(name = "model-loading")]
    ModelLoading,
    /// Test 3: MnistTraining - FC network training on MNIST with encrypted client data
    #[value(name = "mnist-training")]
    MnistTraining,
}

/// MNIST binary format constants (matches download_mnist.py output)
const MNIST_RECORD_SIZE: usize = 785; // 1 byte label + 784 bytes pixel data (28x28)
const MNIST_PIXELS_PER_IMAGE: usize = 28 * 28;
/// Width of the combined tensor: 784 pixel floats + 1 label float
const MNIST_COMBINED_WIDTH: usize = MNIST_PIXELS_PER_IMAGE + 1;

/// Parse MNIST binary format and split across clients.
///
/// MNIST binary format (from download_mnist.py): each record is 785 bytes:
///   byte 0: label (0-9)
///   bytes 1-784: pixel data (28x28, row-major, grayscale)
///
/// Returns: one Vec<f32> per client, storing combined [N, 785] data
/// where each row = [pixel_0, pixel_1, ..., pixel_783, label_as_float]
/// Pixels are normalized to [0,1].
fn parse_mnist_for_clients(
    data: &[u8],
    num_clients: usize,
    images_per_client: usize,
) -> Result<Vec<Vec<f32>>> {
    let total_needed = num_clients * images_per_client;
    let total_records = data.len() / MNIST_RECORD_SIZE;
    if total_records < total_needed {
        return Err(anyhow!(
            "MNIST file has {} images, need {} ({} clients x {})",
            total_records,
            total_needed,
            num_clients,
            images_per_client
        ));
    }

    let mut clients = Vec::with_capacity(num_clients);
    for c in 0..num_clients {
        let start = c * images_per_client;
        let mut combined = Vec::with_capacity(images_per_client * MNIST_COMBINED_WIDTH);

        for i in start..(start + images_per_client) {
            let off = i * MNIST_RECORD_SIZE;
            let label = data[off] as f32;
            let pixels = &data[off + 1..off + 1 + MNIST_PIXELS_PER_IMAGE];

            // Normalize pixels to [0, 1]
            for &px in pixels {
                combined.push(px as f32 / 255.0);
            }
            combined.push(label);
        }
        clients.push(combined);
    }
    Ok(clients)
}

/// Decrypts a WriteRequest using the symmetric key obtained from KMS.ReleaseResults.
///
/// The WriteRequest contains:
/// - data: encrypted ciphertext
/// - first_request_metadata: BlobMetadata with HpkePlusAeadMetadata (contains AAD)
/// - release_token: used to get symmetric_key from KMS.ReleaseResults
///
/// KMS.ReleaseResults returns the unwrapped symmetric key (COSE-encoded).
fn decrypt_write_request(write_req: &WriteRequest, symmetric_key: &[u8]) -> Result<Vec<u8>> {
    let metadata = write_req
        .first_request_metadata
        .as_ref()
        .context("WriteRequest missing first_request_metadata")?;

    let hpke_metadata = match &metadata.encryption_metadata {
        Some(EncryptionMetadata::HpkePlusAeadData(hpke)) => hpke,
        _ => return Err(anyhow!("WriteRequest doesn't have HpkePlusAeadData encryption metadata")),
    };

    // Use decrypt_with_symmetric_key - KMS returns the unwrapped symmetric key directly
    key_derivation::decrypt_with_symmetric_key(
        &write_req.data,                           // ciphertext
        &hpke_metadata.ciphertext_associated_data, // ciphertext AAD
        symmetric_key,                             // COSE-encoded symmetric key from KMS
    )
    .context("Failed to decrypt WriteRequest data")
}

#[derive(Parser, Debug)]
pub struct DemoArgs {
    #[command(flatten)]
    pub launcher_args: LauncherArgs,

    /// Which test to run
    #[arg(long, value_enum, default_value_t = TestType::default())]
    pub test_type: TestType,

    /// KMS address (if not provided, launches KMS using launcher_module)
    #[arg(long)]
    pub kms_address: Option<String>,

    /// KMS bundle path (required if kms_address not provided)
    #[arg(long)]
    pub kms_bundle: Option<PathBuf>,

    /// KMS memory size (e.g., "256M", "1G"). If not set, uses --memory-size.
    #[arg(long)]
    pub kms_memory_size: Option<String>,

    /// KMS ramdrive size in KB. If not set, uses --ramdrive-size.
    #[arg(long)]
    pub kms_ramdrive_size: Option<u32>,

    /// Number of distributed workers to launch (0 = single-node execution)
    #[arg(long, default_value = "0")]
    pub num_workers: u32,

    /// Worker bundle path (required if num_workers > 0)
    #[arg(long)]
    pub worker_bundle: Option<PathBuf>,

    /// Worker memory size (e.g., "256M", "1G"). If not set, uses --memory-size.
    #[arg(long)]
    pub worker_memory_size: Option<String>,

    /// Worker ramdrive size in KB. If not set, uses --ramdrive-size.
    #[arg(long)]
    pub worker_ramdrive_size: Option<u32>,

    /// Path to model zip file (required for model-loading test)
    #[arg(long)]
    pub model_file: Option<PathBuf>,

    /// Path to MNIST binary data file (from download_mnist.py, e.g., train.bin)
    #[arg(long)]
    pub mnist_data_file: Option<PathBuf>,

    /// Path to MNIST test binary file (from download_mnist.py, e.g., test.bin)
    #[arg(long)]
    pub mnist_test_file: Option<PathBuf>,

    /// Number of clients to simulate (default: 5)
    #[arg(long, default_value = "5")]
    pub num_clients: usize,

    /// Number of images per client (default: 100)
    #[arg(long, default_value = "100")]
    pub images_per_client: usize,

    /// Number of FedAvg rounds (default: 3)
    #[arg(long, default_value = "3")]
    pub num_rounds: usize,

    /// Number of local training epochs per client per round (default: 1)
    #[arg(long, default_value = "1")]
    pub local_epochs: usize,

    /// Client learning rate for SGD (default: 0.05)
    #[arg(long, default_value = "0.05")]
    pub learning_rate: f64,

    /// Mini-batch size for local training (default: 64)
    #[arg(long, default_value = "64")]
    pub batch_size: usize,
}

/// Python program for Test 1: ProgramWithDataSource
const PROGRAM_DATA_SOURCE: &str = r#"
import collections
import federated_language
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
from google.protobuf import any_pb2
from fcp.confidentialcompute.python import min_sep_data_source

def trusted_program(input_provider, external_service_handle):

  data_source = min_sep_data_source.MinSepDataSource(
      min_sep=2,
      input_provider=input_provider,
      computation_type=computation_pb2.Type(
          tensor=computation_pb2.TensorType(
              dtype=data_type_pb2.DataType.DT_INT32,
              dims=[3],
          )
      ),
  )
  data_source_iterator = data_source.iterator()

  client_data_type = federated_language.FederatedType(
      federated_language.TensorType(np.int32, [3]), federated_language.CLIENTS
  )

  server_data_type = federated_language.FederatedType(
      federated_language.StructType([
          ('sum', federated_language.TensorType(np.int32, [3])),
          ('client_count', federated_language.TensorType(np.int32, [])),
      ]),
      federated_language.SERVER,
  )

  @tff.tensorflow.computation
  def add(x, y):
    return x + y

  @federated_language.federated_computation(server_data_type, client_data_type)
  def my_comp(server_state, client_data):
    summed_client_data = federated_language.federated_sum(client_data)
    client_count = federated_language.federated_sum(
        federated_language.federated_value(1, federated_language.CLIENTS)
    )
    return tff.learning.templates.LearningProcessOutput(
        federated_language.federated_zip(
            collections.OrderedDict(
                sum=federated_language.federated_map(
                    add, (server_state.sum, summed_client_data)
                ),
                client_count=federated_language.federated_map(
                    add, (server_state.client_count, client_count)
                ),
            )
        ),
        client_count,
    )

  # Run four rounds, which will guarantee that each client is used exactly twice.
  server_state = {'sum': [0, 0, 0], 'client_count': 0}
  for _ in range(4):
    server_state, metrics = my_comp(server_state, data_source_iterator.select(2))

  sum_val, _ = tff.framework.serialize_value(
      server_state["sum"],
      federated_language.framework.infer_type(server_state["sum"]),
  )
  client_count_val, _ = tff.framework.serialize_value(
      server_state["client_count"],
      federated_language.framework.infer_type(server_state["client_count"]),
  )

  external_service_handle.release_unencrypted(
      sum_val.SerializeToString(), b"resulting_sum"
  )
  external_service_handle.release_unencrypted(
      client_count_val.SerializeToString(), b"resulting_client_count"
  )
"#;

/// Python program for Test 2: ProgramWithModelLoading
const PROGRAM_MODEL_LOADING: &str = r#"
import os
import zipfile

import federated_language
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np

def trusted_program(input_provider, external_service_handle):
  zip_file_path = input_provider.get_filename_for_config_id('model1')
  model_path = os.path.join(os.path.dirname(zip_file_path), 'model1')
  with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(model_path)
  model = tff.learning.models.load_functional_model(model_path)

  def model_fn() -> tff.learning.models.VariableModel:
    return tff.learning.models.model_from_functional(model)

  learning_process = tff.learning.algorithms.build_weighted_fed_avg(
      model_fn=model_fn,
      client_optimizer_fn=tff.learning.optimizers.build_sgdm(
          learning_rate=0.01
      ),
  )
  state = learning_process.initialize()

  state_val, _ = tff.framework.serialize_value(
      state,
      federated_language.framework.infer_type(
          state,
      ),
  )
  external_service_handle.release_unencrypted(
      state_val.SerializeToString(), b"result"
  )
"#;

/// Python program for Test 3: MNIST Federated Training
/// Uses MinSepDataSource with custom federated computation for FedAvg.
/// Model: Flatten(784) -> Dense(10, softmax) (7850 params)
/// Loss: categorical cross-entropy. Training with mini-batch SGD.
///
/// Template parameters replaced at runtime:
///   {num_rounds}, {num_clients}, {images_per_client},
///   {local_epochs}, {learning_rate}, {batch_size}
const PROGRAM_MNIST_TRAINING_TEMPLATE: &str = r#"
import federated_language
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
import time
from fcp.confidentialcompute.python import min_sep_data_source

NUM_ROUNDS = {num_rounds}
NUM_CLIENTS = {num_clients}
IMAGES_PER_CLIENT = {images_per_client}
LOCAL_EPOCHS = {local_epochs}
LR = {learning_rate}
BATCH_SIZE = {batch_size}
COMBINED_WIDTH = 785
TOTAL_PARAMS = 7850

def trusted_program(input_provider, external_service_handle):

  # MinSepDataSource: each client tensor is [IMAGES_PER_CLIENT, 785] float32
  data_source = min_sep_data_source.MinSepDataSource(
      min_sep=1,
      input_provider=input_provider,
      computation_type=computation_pb2.Type(
          tensor=computation_pb2.TensorType(
              dtype=data_type_pb2.DataType.DT_FLOAT,
              dims=[IMAGES_PER_CLIENT, COMBINED_WIDTH],
          )
      ),
  )
  data_source_iterator = data_source.iterator()

  client_data_type = federated_language.FederatedType(
      federated_language.TensorType(np.float32, [IMAGES_PER_CLIENT, COMBINED_WIDTH]),
      federated_language.CLIENTS,
  )
  weights_type = federated_language.TensorType(np.float32, [TOTAL_PARAMS])
  server_weights_type = federated_language.FederatedType(
      weights_type, federated_language.SERVER,
  )

  # Weight layout: kernel(7840) bias(10)
  SIZES = [7840, 10]
  SHAPES = [(784, 10), (10,)]

  def _forward(w, client_data):
    """Linear model: Flatten(784) -> Dense(10)."""
    images = client_data[:, :784]
    labels = tf.cast(client_data[:, 784], tf.int32)
    logits = tf.matmul(images, w[0]) + w[1]
    return logits, labels

  # Compute number of batches at Python trace time (IMAGES_PER_CLIENT is a constant)
  NUM_BATCHES = (IMAGES_PER_CLIENT + BATCH_SIZE - 1) // BATCH_SIZE

  @tff.tensorflow.computation
  def client_train(global_weights_flat, client_data):
    w_flat = global_weights_flat
    for _ in range(LOCAL_EPOCHS):
      indices = tf.random.shuffle(tf.range(IMAGES_PER_CLIENT))
      shuffled = tf.gather(client_data, indices)
      for b in range(NUM_BATCHES):
        start = b * BATCH_SIZE
        batch = shuffled[start:start + BATCH_SIZE]
        parts = tf.split(w_flat, SIZES)
        w = [tf.reshape(p, s) for p, s in zip(parts, SHAPES)]
        logits, labels = _forward(w, batch)
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(labels, 10), logits=logits))
        grads = tf.gradients(loss, w)
        updated = [wi - LR * gi for wi, gi in zip(w, grads)]
        w_flat = tf.concat([tf.reshape(u, [-1]) for u in updated], axis=0)
    return w_flat

  @federated_language.federated_computation(server_weights_type, client_data_type)
  def federated_train_round(server_weights, client_data):
    broadcast_weights = federated_language.federated_broadcast(server_weights)
    client_updated = federated_language.federated_map(
        client_train, (broadcast_weights, client_data))
    return federated_language.federated_mean(client_updated)

  def _eval_on_test(w_flat, label):
    """Evaluate model on test set using eager-mode forward pass."""
    test_path = input_provider.get_filename_for_config_id('test_data')
    with open(test_path, 'rb') as f:
      test_bytes = f.read()
    num_test = len(test_bytes) // (COMBINED_WIDTH * 4)
    test_raw = np.frombuffer(test_bytes, dtype=np.float32).reshape(num_test, COMBINED_WIDTH)

    w_np = np.array(w_flat)
    w_list = []
    offset = 0
    for sz, sh in zip(SIZES, SHAPES):
      w_list.append(tf.constant(w_np[offset:offset + sz].reshape(sh)))
      offset += sz

    logits, labels = _forward(w_list, tf.constant(test_raw))
    preds = tf.argmax(logits, axis=1, output_type=tf.int32).numpy()
    labels_np = labels.numpy()
    correct = int(np.sum(preds == labels_np))
    total = len(labels_np)
    loss = float(tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels, 10), logits=logits)).numpy())
    print(label + ' - test: ' + str(correct) + '/' + str(total) +
          ' (' + str(round(100.0 * correct / total, 2)) + '%), loss: ' + str(round(loss, 4)))

  # Glorot uniform initialization for kernel, zeros for bias (matches Keras defaults)
  def glorot_uniform(shape):
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, shape).astype(np.float32)

  np.random.seed(42)
  init_parts = []
  for i, (sz, sh) in enumerate(zip(SIZES, SHAPES)):
    if i % 2 == 0:  # kernel
      init_parts.append(glorot_uniform(sh).flatten())
    else:  # bias
      init_parts.append(np.zeros(sz, dtype=np.float32))
  weights = np.concatenate(init_parts)

  # Evaluate before training (baseline with random init)
  _eval_on_test(weights, 'Round 0/' + str(NUM_ROUNDS))

  # Training loop
  for round_num in range(NUM_ROUNDS):
    client_data = data_source_iterator.select(NUM_CLIENTS)
    t0 = time.time()
    weights = federated_train_round(weights, client_data)
    elapsed = time.time() - t0
    _eval_on_test(weights, 'Round ' + str(round_num + 1) + '/' + str(NUM_ROUNDS) +
          ' (' + str(round(elapsed, 1)) + 's)')

  # Release trained weights
  weights_val, _ = tff.framework.serialize_value(
      weights,
      federated_language.framework.infer_type(weights),
  )
  external_service_handle.release_unencrypted(
      weights_val.SerializeToString(), b"trained_model"
  )
  print('Released trained model weights')
"#;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_max_level(tracing::Level::INFO).init();
    let args = DemoArgs::parse();

    info!("=== Program Executor Demo Cluster ===");
    info!("Test type: {:?}", args.test_type);

    // Pre-compute unique virtio_guest_cid values for all VMs
    let base_cid: u32 = 200;
    let kms_cid = base_cid;
    let worker_base_cid = kms_cid + 1;
    let program_executor_cid = worker_base_cid + args.num_workers;

    // ========================================================================
    // Steps 1-3: Launch KMS, Workers, and Program Executor concurrently
    // ========================================================================
    info!("\n--- Steps 1-3: Launch all TEEs concurrently ---");

    // KMS launch future
    let kms_future = {
        let kms_address_opt = args.kms_address.clone();
        let kms_bundle_opt = args.kms_bundle.clone();
        let launcher_args = args.launcher_args.clone();
        let kms_memory_size = args.kms_memory_size.clone();
        let kms_ramdrive_size = args.kms_ramdrive_size;
        let span = info_span!("KMS");
        async move {
            if let Some(addr) = kms_address_opt {
                info!("Using provided KMS address: {}", addr);
                Ok::<_, anyhow::Error>((addr, None))
            } else {
                info!("Launching KMS TEE...");
                let kms_bundle = kms_bundle_opt
                    .context("--kms-bundle required when --kms-address not provided")?;

                let mut kms_qemu_params = launcher_args.qemu_params.clone();
                kms_qemu_params.virtio_guest_cid = Some(kms_cid);
                if let Some(ref mem) = kms_memory_size {
                    kms_qemu_params.memory_size = Some(mem.clone());
                }
                if let Some(ramdrive) = kms_ramdrive_size {
                    kms_qemu_params.ramdrive_size = ramdrive;
                }

                let kms_args = LauncherArgs {
                    system_image: launcher_args.system_image.clone(),
                    container_bundle: kms_bundle,
                    qemu_params: kms_qemu_params,
                    communication_channel: launcher_args.communication_channel.clone(),
                    application_config: launcher_args.application_config.clone(),
                };

                let mut launcher = Launcher::create(kms_args).await?;
                let address = launcher.get_trusted_app_address().await?;
                let addr = format!("http://{}", address);
                info!("KMS TEE launched at: {} (CID: {})", addr, kms_cid);
                Ok((addr, Some(launcher)))
            }
        }.instrument(span)
    };

    // Workers launch future
    let workers_future = {
        let num_workers = args.num_workers;
        let worker_bundle_opt = args.worker_bundle.clone();
        let launcher_args = args.launcher_args.clone();
        let worker_memory_size = args.worker_memory_size.clone();
        let worker_ramdrive_size = args.worker_ramdrive_size;
        let span = info_span!("Workers");
        async move {
            let mut worker_launchers = Vec::new();
            let mut worker_bns_addresses = Vec::new();

            if num_workers > 0 {
                info!("Launching {} Worker TEEs...", num_workers);
                let worker_bundle = worker_bundle_opt
                    .context("--worker-bundle required when --num-workers > 0")?;

                let mut worker_futures = Vec::new();
                for i in 0..num_workers {
                    let cid = worker_base_cid + i;
                    let mut worker_qemu_params = launcher_args.qemu_params.clone();
                    worker_qemu_params.virtio_guest_cid = Some(cid);
                    if let Some(ref mem) = worker_memory_size {
                        worker_qemu_params.memory_size = Some(mem.clone());
                    }
                    if let Some(ramdrive) = worker_ramdrive_size {
                        worker_qemu_params.ramdrive_size = ramdrive;
                    }

                    let worker_args = LauncherArgs {
                        system_image: launcher_args.system_image.clone(),
                        container_bundle: worker_bundle.clone(),
                        qemu_params: worker_qemu_params,
                        communication_channel: launcher_args.communication_channel.clone(),
                        application_config: launcher_args.application_config.clone(),
                    };

                    let worker_span = info_span!("Worker", id = i + 1);
                    worker_futures.push(async move {
                        let mut launcher = Launcher::create(worker_args).await?;
                        let address = launcher.get_trusted_app_address().await?;
                        let worker_addr = format!("http://{}", address);
                        info!("Started at: {} (CID: {})", worker_addr, cid);
                        Ok::<_, anyhow::Error>((worker_addr, launcher))
                    }.instrument(worker_span));
                }

                let results = futures::future::join_all(worker_futures).await;
                for result in results {
                    let (addr, launcher) = result?;
                    worker_bns_addresses.push(addr);
                    worker_launchers.push(launcher);
                }
            } else {
                info!("Single-node execution (no workers)");
            }

            Ok::<_, anyhow::Error>((worker_launchers, worker_bns_addresses))
        }.instrument(span)
    };

    // Program executor launch future
    let program_executor_future = {
        let launcher_args = args.launcher_args.clone();
        let span = info_span!("ProgramExecutor");
        async move {
            info!("Launching Program Executor TEE...");
            let mut qemu_params = launcher_args.qemu_params.clone();
            qemu_params.virtio_guest_cid = Some(program_executor_cid);
            qemu_params.data_service_port = Some(launcher_module::DATA_SERVICE_PORT);

            let executor_args = LauncherArgs {
                system_image: launcher_args.system_image.clone(),
                container_bundle: launcher_args.container_bundle.clone(),
                qemu_params,
                communication_channel: launcher_args.communication_channel.clone(),
                application_config: launcher_args.application_config.clone(),
            };

            let mut launcher = Launcher::create(executor_args).await?;
            let address = launcher.get_trusted_app_address().await?;
            let tee_address = format!("http://{}", address);
            info!("Program executor TEE started at: {} (CID: {})", tee_address, program_executor_cid);
            Ok::<_, anyhow::Error>((tee_address, launcher))
        }.instrument(span)
    };

    // Launch all TEEs concurrently
    let (kms_result, workers_result, executor_result) =
        tokio::try_join!(kms_future, workers_future, program_executor_future)?;

    let (kms_address, kms_launcher) = kms_result;
    let (mut worker_launchers, worker_bns_addresses) = workers_result;
    let (tee_address, mut launcher) = executor_result;

    // ========================================================================
    // Step 4: Get Evidence and Reference Values
    // ========================================================================
    info!("\n--- Step 4: Get Evidence and Reference Values ---");
    let endorsed_evidence = launcher.get_endorsed_evidence().await?;
    let extracted_evidence = extract_evidence(
        endorsed_evidence.evidence.as_ref().context("No evidence in endorsed evidence")?,
    )?;
    let reference_values: ReferenceValues =
        create_reference_values_for_extracted_evidence(extracted_evidence).convert()?;

    // Skip worker attestation: Oak Launcher doesn't populate platform
    // endorsements yet (b/375137648), so passing None makes the root TEE use
    // fake attestation mode, matching the worker's fake fallback in
    // server_session_config.rs.
    let worker_reference_values: Option<ReferenceValues> = None;
    info!("Worker attestation skipped (platform endorsements not available)");

    // ========================================================================
    // Step 5: Select Program and Configure Test Parameters
    // ========================================================================
    info!("\n--- Step 5: Select Program & Configure Test ---");

    // For MnistTraining, we need an owned String for the interpolated program.
    // Other test types use static &str. We handle both via program_owned.
    let mut program_owned: Option<String> = None;
    let mut mnist_client_data: Option<Vec<Vec<f32>>> = None;

    let (program, client_ids, client_data_dir, model_files, result_keys): (
        &str,
        Vec<String>,
        &str,
        Vec<(String, Vec<u8>)>,
        Vec<&str>,
    ) = match args.test_type {
        TestType::DataSource => {
            info!("Running Test 1: ProgramWithDataSource");
            (
                PROGRAM_DATA_SOURCE,
                vec![
                    "client1".to_string(),
                    "client2".to_string(),
                    "client3".to_string(),
                    "client4".to_string(),
                ],
                "data_dir",
                Vec::new(),
                vec!["resulting_sum", "resulting_client_count"],
            )
        }
        TestType::ModelLoading => {
            info!("Running Test 2: ProgramWithModelLoading");
            let model_path =
                args.model_file.as_ref().context("--model-file required for model-loading test")?;
            let model_data = std::fs::read(model_path)
                .with_context(|| format!("Failed to read model file: {:?}", model_path))?;
            info!("Loaded model file: {:?} ({} bytes)", model_path, model_data.len());
            (
                PROGRAM_MODEL_LOADING,
                Vec::new(), // No client data
                "",         // No client data dir
                vec![("model1".to_string(), model_data)],
                vec!["result"],
            )
        }
        TestType::MnistTraining => {
            info!("Running Test 3: MNIST Federated Training (MinSepDataSource + FedAvg)");

            // Load and parse MNIST training data
            let mnist_path = args
                .mnist_data_file
                .as_ref()
                .context("--mnist-data-file required for mnist-training test")?;
            let mnist_raw = std::fs::read(mnist_path)
                .with_context(|| format!("Failed to read MNIST file: {:?}", mnist_path))?;
            info!(
                "Loaded MNIST train file: {:?} ({} bytes, {} records)",
                mnist_path,
                mnist_raw.len(),
                mnist_raw.len() / MNIST_RECORD_SIZE
            );

            let clients_data =
                parse_mnist_for_clients(&mnist_raw, args.num_clients, args.images_per_client)?;
            info!(
                "Parsed MNIST: {} clients x {} images each",
                args.num_clients, args.images_per_client
            );

            // Load and parse MNIST test data for evaluation
            let test_path = args
                .mnist_test_file
                .as_ref()
                .context("--mnist-test-file required for mnist-training test")?;
            let test_raw = std::fs::read(test_path)
                .with_context(|| format!("Failed to read MNIST test file: {:?}", test_path))?;
            let num_test_images = test_raw.len() / MNIST_RECORD_SIZE;
            // Reuse the same parsing (normalizes to [0,1], label appended)
            let test_floats = parse_mnist_for_clients(&test_raw, 1, num_test_images)?;
            let test_data_bytes: Vec<u8> =
                test_floats[0].iter().flat_map(|f| f.to_le_bytes()).collect();
            info!(
                "Loaded MNIST test file: {:?} ({} images, {} bytes as f32)",
                test_path,
                num_test_images,
                test_data_bytes.len()
            );

            let client_ids: Vec<String> =
                (0..args.num_clients).map(|i| format!("client{}", i)).collect();

            // Interpolate runtime values into the Python program template
            let prog = PROGRAM_MNIST_TRAINING_TEMPLATE
                .replace("{num_rounds}", &args.num_rounds.to_string())
                .replace("{num_clients}", &args.num_clients.to_string())
                .replace("{images_per_client}", &args.images_per_client.to_string())
                .replace("{local_epochs}", &args.local_epochs.to_string())
                .replace("{learning_rate}", &args.learning_rate.to_string())
                .replace("{batch_size}", &args.batch_size.to_string());
            program_owned = Some(prog);
            mnist_client_data = Some(clients_data);

            (
                // Will be replaced below with &program_owned
                "",
                client_ids,
                "mnist_data",
                vec![("test_data".to_string(), test_data_bytes)],
                vec!["trained_model"],
            )
        }
    };

    // Use the owned program string for MnistTraining
    let program = if let Some(ref owned) = program_owned { owned.as_str() } else { program };

    // ========================================================================
    // Step 6: Connect to KMS and Compute Policy Hash
    // ========================================================================
    info!("\n--- Step 6: Connect to KMS & Compute Policy Hash ---");
    let mut kms_client = KmsClient::new(&kms_address).await?;
    kms_client.rotate_keyset(1, 7200).await?;

    let logical_pipeline_name = "program_executor_pipeline".to_string();
    let access_policy =
        create_program_executor_access_policy(Some(reference_values.clone()), program);
    let policy_hash = compute_policy_hash(&access_policy.encode_to_vec());
    info!("Computed policy hash: {} bytes", policy_hash.len());

    // Extract the PipelineVariantPolicy from the DataAccessPolicy
    let variant_policy = access_policy
        .pipelines
        .get(&logical_pipeline_name)
        .and_then(|logical_policy| logical_policy.instances.first().cloned())
        .context("No pipeline variant found in access policy")?;
    let variant_policy_bytes = variant_policy.encode_to_vec();
    info!("Extracted variant policy: {} bytes", variant_policy_bytes.len());

    let public_keys = kms_client.derive_keys(1, vec![policy_hash.clone()]).await?;
    let kms_public_key_cwt =
        public_keys.into_iter().next().context("No public key returned from derive_keys")?;
    info!("Received KMS public key CWT: {} bytes", kms_public_key_cwt.len());

    // Extract the COSE key from the CWT (KMS returns keys wrapped in CWTs)
    let kms_public_key = key_derivation::extract_key_bytes_from_cwt(&kms_public_key_cwt)
        .context("Failed to extract COSE key from CWT")?;
    info!("Extracted COSE key: {} bytes", kms_public_key.len());

    // ========================================================================
    // Step 7: Start RealDataReadWriteService with KMS Public Key
    // ========================================================================
    info!("\n--- Step 7: Start RealDataReadWriteService ---");
    let data_service = Arc::new(
        RealDataReadWriteService::new_with_kms_key(kms_public_key, policy_hash.clone())
            .context("Failed to create RealDataReadWriteService")?,
    );

    // Start on the fixed port that QEMU guestfwd is configured for
    let data_service_port = launcher_module::DATA_SERVICE_PORT;

    // Build the gRPC server with DataReadWrite, and optionally ComputationDelegation proxy
    {
        use data_read_write_proto::fcp::confidentialcompute::outgoing::data_read_write_server::DataReadWriteServer;
        use computation_delegation_proto::fcp::confidentialcompute::outgoing::computation_delegation_server::ComputationDelegationServer;
        use std::net::Ipv4Addr;

        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, data_service_port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        let data_rw_svc = DataReadWriteServer::from_arc(data_service.clone());

        if !worker_bns_addresses.is_empty() {
            let proxy = computation_delegation_proxy::ComputationDelegationProxy::new(
                worker_bns_addresses.clone(),
            )
            .await
            .map_err(|e| anyhow!("Failed to create ComputationDelegationProxy: {}", e))?;
            // Match the 2GB limit used by the root TEE's gRPC channel.
            const MAX_GRPC_MSG: usize = 2 * 1000 * 1000 * 1000;
            let delegation_svc = ComputationDelegationServer::new(proxy)
                .max_decoding_message_size(MAX_GRPC_MSG)
                .max_encoding_message_size(MAX_GRPC_MSG);

            info!("Starting combined DataReadWrite + ComputationDelegation server on port {}", data_service_port);
            tokio::spawn(async move {
                tonic::transport::Server::builder()
                    .add_service(data_rw_svc)
                    .add_service(delegation_svc)
                    .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                    .await
                    .expect("gRPC server error");
            });
        } else {
            tokio::spawn(async move {
                tonic::transport::Server::builder()
                    .add_service(data_rw_svc)
                    .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                    .await
                    .expect("gRPC server error");
            });
        }
    }

    // The TEE will connect to this address (forwarded by QEMU guestfwd)
    let data_service_address = launcher_module::VM_DATA_SERVICE_ADDRESS.to_string();
    info!(
        "Outgoing server listening at: 127.0.0.1:{} (VM accessible at {})",
        data_service_port, data_service_address
    );

    // ========================================================================
    // Step 8: Store Client Test Data
    // ========================================================================
    match args.test_type {
        TestType::DataSource => {
            info!("\n--- Step 8: Store Client Test Data (DataSource INT32) ---");
            let tensor_name = "output_tensor_name";

            for (i, client_id) in client_ids.iter().enumerate() {
                let uri = format!("{}/{}", client_data_dir, client_id);
                let values = vec![(1 + i * 3) as i32, (2 + i * 3) as i32, (3 + i * 3) as i32];

                let checkpoint = checkpoint_rs::build_checkpoint_from_ints(&values, tensor_name)
                    .map_err(|e| anyhow::anyhow!("Failed to build checkpoint: {}", e))?;

                data_service.store_encrypted_message_for_kms(&uri, &checkpoint, None)?;
                info!("Stored test data for {} ({} bytes)", client_id, checkpoint.len());
            }
        }
        TestType::MnistTraining => {
            info!("\n--- Step 8: Store MNIST Client Data (float32 checkpoints) ---");
            let tensor_name = "output_tensor_name";
            let clients_data =
                mnist_client_data.as_ref().context("MNIST client data not parsed")?;

            for (i, client_id) in client_ids.iter().enumerate() {
                let uri = format!("{}/{}", client_data_dir, client_id);
                let client_floats = &clients_data[i];
                let num_images = args.images_per_client as i32;

                // Shape: [N, 785] - combined images + labels
                let shape = [num_images, MNIST_COMBINED_WIDTH as i32];
                let checkpoint =
                    checkpoint_rs::build_checkpoint_from_floats(client_floats, &shape, tensor_name)
                        .map_err(|e| anyhow::anyhow!("Failed to build MNIST checkpoint: {}", e))?;

                data_service.store_encrypted_message_for_kms(&uri, &checkpoint, None)?;
                info!(
                    "Stored MNIST data for {} ({} images, {} bytes checkpoint)",
                    client_id,
                    num_images,
                    checkpoint.len()
                );
            }
        }
        TestType::ModelLoading => {
            info!("\n--- Step 8: No client data (model loading test) ---");
        }
    }

    // ========================================================================
    // Step 9: Register Pipeline and Authorize
    // ========================================================================
    info!("\n--- Step 9: Register Pipeline and Authorize ---");

    let (invocation_id, _) = kms_client
        .register_pipeline_invocation(
            logical_pipeline_name,
            variant_policy_bytes.clone(),
            vec![1],
            vec![access_policy.encode_to_vec()],
            3600,
        )
        .await?;

    let launcher_evidence = endorsed_evidence.evidence.as_ref().and_then(|e| e.convert().ok());
    let launcher_endorsements =
        endorsed_evidence.endorsements.as_ref().and_then(|e| e.convert().ok());

    let (protected_response, signing_key_endorsement) = kms_client
        .authorize_transform(
            invocation_id,
            variant_policy_bytes,
            launcher_evidence,
            launcher_endorsements,
        )
        .await?;

    // ========================================================================
    // Step 10: Initialize Program Executor with Python Program
    // ========================================================================
    info!("\n--- Step 10: Initialize Program Executor ---");
    let mut program_client = ProgramExecutorClient::new(&tee_address).await?;
    program_client
        .initialize_with_program(
            protected_response,
            program,
            client_ids,
            client_data_dir,
            &data_service_address,
            worker_bns_addresses,
            worker_reference_values,
            Vec::new(), // No private state needed when using KMS
            model_files,
        )
        .await?;

    // ========================================================================
    // Step 11: Execute Session
    // ========================================================================
    info!("\n--- Step 11: Execute Python Program ---");
    let (_results, _release_token) = program_client.execute_session().await?;

    // ========================================================================
    // Step 12: Collect WriteRequests (still encrypted)
    // ========================================================================
    info!("\n--- Step 12: Collect WriteRequests ---");

    let mut write_requests = Vec::new();
    for key in &result_keys {
        let write_req = data_service
            .get_write_request_for_key(key)
            .with_context(|| format!("Failed to get WriteRequest for '{}'", key))?;
        info!(
            "Got WriteRequest for '{}': {} bytes encrypted, {} bytes release_token",
            key,
            write_req.data.len(),
            write_req.release_token.len()
        );
        write_requests.push((key.to_string(), write_req));
    }

    // ========================================================================
    // Step 13: Release with KMS to get decryption keys
    // ========================================================================
    info!("\n--- Step 13: Release Results via KMS ---");

    // Collect all release_tokens with signing_key_endorsement
    let release_requests: Vec<_> = write_requests
        .iter()
        .map(|(_, write_req)| (write_req.release_token.clone(), signing_key_endorsement.clone()))
        .collect();

    let symmetric_keys = kms_client.release_results(release_requests).await?;
    info!("Received {} symmetric keys from KMS", symmetric_keys.len());

    // ========================================================================
    // Step 14: Decrypt and Verify Results
    // ========================================================================
    info!("\n--- Step 14: Decrypt and Verify Results ---");

    for (i, (key, write_req)) in write_requests.iter().enumerate() {
        let symmetric_key =
            symmetric_keys.get(i).with_context(|| format!("Missing symmetric key for {}", key))?;
        let decrypted_data = decrypt_write_request(write_req, symmetric_key)
            .with_context(|| format!("Failed to decrypt {}", key))?;

        info!("Result '{}' from TEE: {} bytes", key, decrypted_data.len());
        // info!("  -> Raw data: {:?}", decrypted_data);

        // Print expected values hint based on test type
        match args.test_type {
            TestType::DataSource => {
                if key == &"resulting_sum" {
                    info!("  -> Expected: TFF Value with array [44, 52, 60]");
                } else if key == &"resulting_client_count" {
                    info!("  -> Expected: TFF Value with array [8]");
                }
            }
            TestType::ModelLoading => {
                info!("  -> Expected: TFF Value struct with 5 elements");
            }
            TestType::MnistTraining => {
                let output_path = "/tmp/mnist_trained_model.bin";
                std::fs::write(output_path, &decrypted_data)
                    .with_context(|| format!("Failed to write weights to {}", output_path))?;
                info!(
                    "  -> Trained model weights ({} rounds, {} clients x {} images)",
                    args.num_rounds, args.num_clients, args.images_per_client
                );
                info!("  -> Saved to: {}", output_path);
            }
        }
    }

    // ========================================================================
    // Step 15: Cleanup
    // ========================================================================
    info!("\n--- Step 15: Cleanup ---");
    for mut worker in worker_launchers {
        let _ = worker.kill().await;
    }
    if let Some(mut kms) = kms_launcher {
        let _ = kms.kill().await;
    }
    let _ = launcher.kill().await;

    info!("\n=== Demo Complete ===");
    Ok(())
}
