use anyhow::{anyhow, Context, Result};
use clap::{Parser, ValueEnum};
use prost::Message;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

mod attestation_factory;
mod client_simulator;
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
    /// Test 3: CifarTraining - CNN training on CIFAR-10 with encrypted client data
    #[value(name = "cifar-training")]
    CifarTraining,
}

/// CIFAR-10 binary format constants
const CIFAR10_RECORD_SIZE: usize = 3073; // 1 byte label + 3072 bytes pixel data
const CIFAR10_PIXELS_PER_IMAGE: usize = 32 * 32 * 3;
/// Width of the combined tensor: 3072 pixel floats + 1 label float
const CIFAR10_COMBINED_WIDTH: usize = CIFAR10_PIXELS_PER_IMAGE + 1;

/// Parse CIFAR-10 binary format and split across clients.
///
/// CIFAR-10 binary format: each record is 3073 bytes:
///   byte 0: label (0-9)
///   bytes 1-3072: pixel data in channels-first (R,R,...,G,G,...,B,B,...)
///
/// Returns: one Vec<f32> per client, storing combined [N, 3073] data
/// where each row = [pixel_r, pixel_g, pixel_b, ..., label_as_float]
/// Pixels are converted to channels-last (HWC) and normalized to [0,1].
fn parse_cifar10_for_clients(
    data: &[u8],
    num_clients: usize,
    images_per_client: usize,
) -> Result<Vec<Vec<f32>>> {
    let total_needed = num_clients * images_per_client;
    let total_records = data.len() / CIFAR10_RECORD_SIZE;
    if total_records < total_needed {
        return Err(anyhow!(
            "CIFAR-10 file has {} images, need {} ({} clients x {})",
            total_records,
            total_needed,
            num_clients,
            images_per_client
        ));
    }

    let mut clients = Vec::with_capacity(num_clients);
    for c in 0..num_clients {
        let start = c * images_per_client;
        let mut combined = Vec::with_capacity(images_per_client * CIFAR10_COMBINED_WIDTH);

        for i in start..(start + images_per_client) {
            let off = i * CIFAR10_RECORD_SIZE;
            let label = data[off] as f32;
            let pixels = &data[off + 1..off + 1 + CIFAR10_PIXELS_PER_IMAGE];

            // Convert channels-first (RRR...GGG...BBB...) -> channels-last (RGBRGB...)
            for px in 0..(32 * 32) {
                combined.push(pixels[px] as f32 / 255.0); // R
                combined.push(pixels[1024 + px] as f32 / 255.0); // G
                combined.push(pixels[2048 + px] as f32 / 255.0); // B
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

    /// Path to CIFAR-10 binary data file (e.g., data_batch_1.bin)
    #[arg(long)]
    pub cifar_data_file: Option<PathBuf>,

    /// Number of CIFAR-10 clients to simulate (default: 2)
    #[arg(long, default_value = "2")]
    pub num_cifar_clients: usize,

    /// Number of images per client for CIFAR-10 test (default: 100)
    #[arg(long, default_value = "100")]
    pub images_per_client: usize,

    /// Number of FedAvg rounds for CIFAR-10 test (default: 2)
    #[arg(long, default_value = "2")]
    pub num_rounds: usize,
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

/// Python program for Test 3: CIFAR-10 Federated Training
/// Uses MinSepDataSource with custom federated computation for FedAvg.
/// CNN: Conv(6,5)->Pool->Conv(16,5)->Pool->FC(120)->FC(84)->FC(10) (62006 params)
/// Training uses pure tensor ops inside @tff.tensorflow.computation (graph-safe).
///
/// {num_rounds}, {num_cifar_clients}, {images_per_client} are replaced at runtime.
const PROGRAM_CIFAR_TRAINING_TEMPLATE: &str = r#"
import federated_language
from federated_language.proto import computation_pb2
from federated_language.proto import data_type_pb2
import tensorflow_federated as tff
import tensorflow as tf
import numpy as np
from fcp.confidentialcompute.python import min_sep_data_source

NUM_ROUNDS = {num_rounds}
NUM_CLIENTS = {num_cifar_clients}
IMAGES_PER_CLIENT = {images_per_client}
COMBINED_WIDTH = 3073
TOTAL_PARAMS = 62006

def trusted_program(input_provider, external_service_handle):

  # MinSepDataSource: each client tensor is [IMAGES_PER_CLIENT, 3073] float32
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

  # Weight layout: conv1_k(450) conv1_b(6) conv2_k(2400) conv2_b(16)
  #   fc1_k(48000) fc1_b(120) fc2_k(10080) fc2_b(84) fc3_k(840) fc3_b(10)
  SIZES = [450, 6, 2400, 16, 48000, 120, 10080, 84, 840, 10]
  SHAPES = [
      (5,5,3,6), (6,), (5,5,6,16), (16,),
      (400,120), (120,), (120,84), (84,), (84,10), (10,),
  ]

  @tff.tensorflow.computation
  def client_train(global_weights_flat, client_data):
    parts = tf.split(global_weights_flat, SIZES)
    w = [tf.reshape(p, s) for p, s in zip(parts, SHAPES)]

    images = tf.reshape(client_data[:, :3072], [-1, 32, 32, 3])
    labels = tf.cast(client_data[:, 3072], tf.int32)

    # Forward: Conv(6,5)->Pool->Conv(16,5)->Pool->FC(120)->FC(84)->FC(10)
    x = tf.nn.relu(tf.nn.conv2d(images, w[0], strides=1, padding='VALID') + w[1])
    x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
    x = tf.nn.relu(tf.nn.conv2d(x, w[2], strides=1, padding='VALID') + w[3])
    x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='VALID')
    x = tf.reshape(x, [-1, 400])
    x = tf.nn.relu(tf.matmul(x, w[4]) + w[5])
    x = tf.nn.relu(tf.matmul(x, w[6]) + w[7])
    logits = tf.matmul(x, w[8]) + w[9]

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    grads = tf.gradients(loss, w)
    updated = [wi - 0.01 * gi for wi, gi in zip(w, grads)]
    return tf.concat([tf.reshape(u, [-1]) for u in updated], axis=0)

  @federated_language.federated_computation(server_weights_type, client_data_type)
  def federated_train_round(server_weights, client_data):
    broadcast_weights = federated_language.federated_broadcast(server_weights)
    client_updated = federated_language.federated_map(
        client_train, (broadcast_weights, client_data))
    return federated_language.federated_mean(client_updated)

  weights = np.zeros(TOTAL_PARAMS, dtype=np.float32)
  for round_num in range(NUM_ROUNDS):
    client_data = data_source_iterator.select(NUM_CLIENTS)
    weights = federated_train_round(weights, client_data)
    print('Completed round ' + str(round_num + 1) + '/' + str(NUM_ROUNDS))

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

    // Counter for unique virtio_guest_cid values
    // Each VM needs a unique CID on the host
    let base_cid: u32 = 100;
    let mut next_cid = base_cid;

    // ========================================================================
    // Step 1: Launch or Connect to KMS
    // ========================================================================
    info!("\n--- Step 1: Launch/Connect KMS ---");
    let (kms_address, kms_launcher) = if let Some(addr) = args.kms_address.clone() {
        info!("Using provided KMS address: {}", addr);
        (addr, None)
    } else {
        info!("No KMS address provided, launching KMS TEE...");
        let kms_bundle = args
            .kms_bundle
            .clone()
            .context("--kms-bundle required when --kms-address not provided")?;

        let mut kms_qemu_params = args.launcher_args.qemu_params.clone();
        kms_qemu_params.virtio_guest_cid = Some(next_cid);
        next_cid += 1;
        // Override memory and ramdrive size if KMS-specific values provided
        if let Some(ref mem) = args.kms_memory_size {
            kms_qemu_params.memory_size = Some(mem.clone());
        }
        if let Some(ramdrive) = args.kms_ramdrive_size {
            kms_qemu_params.ramdrive_size = ramdrive;
        }

        let kms_args = LauncherArgs {
            system_image: args.launcher_args.system_image.clone(),
            container_bundle: kms_bundle,
            qemu_params: kms_qemu_params,
            communication_channel: args.launcher_args.communication_channel.clone(),
            application_config: args.launcher_args.application_config.clone(),
        };

        let mut launcher = Launcher::create(kms_args).await?;
        let address = launcher.get_trusted_app_address().await?;
        let addr = format!("http://{}", address);
        info!("KMS TEE launched at: {} (CID: {})", addr, next_cid - 1);
        (addr, Some(launcher))
    };

    // ========================================================================
    // Step 2: Launch Worker TEEs (if distributed execution requested)
    // ========================================================================
    let mut worker_launchers = Vec::new();
    let mut worker_bns_addresses = Vec::new();

    if args.num_workers > 0 {
        info!("\n--- Step 2: Launch {} Worker TEEs ---", args.num_workers);
        let worker_bundle = args
            .worker_bundle
            .clone()
            .context("--worker-bundle required when --num-workers > 0")?;

        for i in 0..args.num_workers {
            info!("Launching worker {} of {}...", i + 1, args.num_workers);

            let mut worker_qemu_params = args.launcher_args.qemu_params.clone();
            worker_qemu_params.virtio_guest_cid = Some(next_cid);
            next_cid += 1;
            // Override memory and ramdrive size if worker-specific values provided
            if let Some(ref mem) = args.worker_memory_size {
                worker_qemu_params.memory_size = Some(mem.clone());
            }
            if let Some(ramdrive) = args.worker_ramdrive_size {
                worker_qemu_params.ramdrive_size = ramdrive;
            }

            let worker_args = LauncherArgs {
                system_image: args.launcher_args.system_image.clone(),
                container_bundle: worker_bundle.clone(),
                qemu_params: worker_qemu_params,
                communication_channel: args.launcher_args.communication_channel.clone(),
                application_config: args.launcher_args.application_config.clone(),
            };

            let mut launcher = Launcher::create(worker_args).await?;
            let address = launcher.get_trusted_app_address().await?;
            let worker_addr = format!("http://{}", address);
            info!("Worker {} started at: {} (CID: {})", i + 1, worker_addr, next_cid - 1);

            worker_bns_addresses.push(worker_addr);
            worker_launchers.push(launcher);
        }
    } else {
        info!("\n--- Step 2: Single-node execution (no workers) ---");
    }

    // ========================================================================
    // Step 3: Launch Program Executor TEE
    // ========================================================================
    info!("\n--- Step 3: Launch Program Executor TEE ---");
    let mut program_executor_qemu_params = args.launcher_args.qemu_params.clone();
    program_executor_qemu_params.virtio_guest_cid = Some(next_cid);
    // Set fixed data service port for QEMU guestfwd (so TEE can reach host's data service)
    program_executor_qemu_params.data_service_port = Some(launcher_module::DATA_SERVICE_PORT);
    let program_executor_cid = next_cid;
    next_cid += 1;
    let _ = next_cid; // Suppress unused warning

    let program_executor_args = LauncherArgs {
        system_image: args.launcher_args.system_image.clone(),
        container_bundle: args.launcher_args.container_bundle.clone(),
        qemu_params: program_executor_qemu_params,
        communication_channel: args.launcher_args.communication_channel.clone(),
        application_config: args.launcher_args.application_config.clone(),
    };

    let mut launcher = Launcher::create(program_executor_args).await?;
    let trusted_app_address = launcher.get_trusted_app_address().await?;
    let tee_address = format!("http://{}", trusted_app_address);
    info!("Program executor TEE started at: {} (CID: {})", tee_address, program_executor_cid);

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

    let worker_reference_values = if !worker_launchers.is_empty() {
        let worker_evidence = worker_launchers[0].get_endorsed_evidence().await?;
        let extracted_worker_evidence = extract_evidence(
            worker_evidence.evidence.as_ref().context("No evidence in worker endorsed evidence")?,
        )?;
        Some(create_reference_values_for_extracted_evidence(extracted_worker_evidence).convert()?)
    } else {
        None
    };
    info!("Extracted reference values for main TEE and workers (if any)");

    // ========================================================================
    // Step 5: Select Program and Configure Test Parameters
    // ========================================================================
    info!("\n--- Step 5: Select Program & Configure Test ---");

    // For CifarTraining, we need an owned String for the interpolated program.
    // Other test types use static &str. We handle both via program_owned.
    let mut program_owned: Option<String> = None;
    let mut cifar_client_data: Option<Vec<Vec<f32>>> = None;

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
        TestType::CifarTraining => {
            info!("Running Test 3: CIFAR-10 Federated Training (MinSepDataSource + FedAvg)");

            // Load and parse CIFAR-10 data
            let cifar_path = args
                .cifar_data_file
                .as_ref()
                .context("--cifar-data-file required for cifar-training test")?;
            let cifar_raw = std::fs::read(cifar_path)
                .with_context(|| format!("Failed to read CIFAR-10 file: {:?}", cifar_path))?;
            info!(
                "Loaded CIFAR-10 file: {:?} ({} bytes, {} records)",
                cifar_path,
                cifar_raw.len(),
                cifar_raw.len() / CIFAR10_RECORD_SIZE
            );

            let clients_data = parse_cifar10_for_clients(
                &cifar_raw,
                args.num_cifar_clients,
                args.images_per_client,
            )?;
            info!(
                "Parsed CIFAR-10: {} clients x {} images each",
                args.num_cifar_clients, args.images_per_client
            );

            let client_ids: Vec<String> =
                (0..args.num_cifar_clients).map(|i| format!("client{}", i)).collect();

            // Interpolate runtime values into the Python program template
            let prog = PROGRAM_CIFAR_TRAINING_TEMPLATE
                .replace("{num_rounds}", &args.num_rounds.to_string())
                .replace("{num_cifar_clients}", &args.num_cifar_clients.to_string())
                .replace("{images_per_client}", &args.images_per_client.to_string());
            program_owned = Some(prog);
            cifar_client_data = Some(clients_data);

            (
                // Will be replaced below with &program_owned
                "",
                client_ids,
                "cifar_data",
                Vec::new(), // No model files needed (model built inline)
                vec!["trained_model"],
            )
        }
    };

    // Use the owned program string for CifarTraining
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
    data_service.clone().start_server_on_port(data_service_port).await?;
    // The TEE will connect to this address (forwarded by QEMU guestfwd)
    let data_service_address = launcher_module::VM_DATA_SERVICE_ADDRESS.to_string();
    info!(
        "RealDataReadWriteService listening at: 127.0.0.1:{} (VM accessible at {})",
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
        TestType::CifarTraining => {
            info!("\n--- Step 8: Store CIFAR-10 Client Data (float32 checkpoints) ---");
            let tensor_name = "output_tensor_name";
            let clients_data =
                cifar_client_data.as_ref().context("CIFAR client data not parsed")?;

            for (i, client_id) in client_ids.iter().enumerate() {
                let uri = format!("{}/{}", client_data_dir, client_id);
                let client_floats = &clients_data[i];
                let num_images = args.images_per_client as i32;

                // Shape: [N, 3073] - combined images + labels
                let shape = [num_images, CIFAR10_COMBINED_WIDTH as i32];
                let checkpoint =
                    checkpoint_rs::build_checkpoint_from_floats(client_floats, &shape, tensor_name)
                        .map_err(|e| anyhow::anyhow!("Failed to build CIFAR checkpoint: {}", e))?;

                data_service.store_encrypted_message_for_kms(&uri, &checkpoint, None)?;
                info!(
                    "Stored CIFAR-10 data for {} ({} images, {} bytes checkpoint)",
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
            TestType::CifarTraining => {
                info!(
                    "  -> Trained model weights ({} rounds, {} clients x {} images)",
                    args.num_rounds, args.num_cifar_clients, args.images_per_client
                );
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
