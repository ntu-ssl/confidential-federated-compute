QEMU_PATH="/tmp2/r14922032/qemu-v10.1.0/build/qemu-system-x86_64"

ARTIFACTS_DIR="/home/r14922032/cfc_tensorflow/artifacts"

LAUNCHER_EXEC_PATH="$ARTIFACTS_DIR/demo_cluster"

OAK_STAGE0_DEST="stage0_bin"
OAK_KERNEL_DEST="oak_containers_kernel"
OAK_STAGE1_DEST="oak_containers_stage1.cpio"
OAK_SYSTEM_IMAGE_DEST="oak_containers_system_image.tar.xz"

KMS_RAMDRIVE_SIZE="1000000" # 10MB
KMS_MEMORY_SIZE="2G"

PROGRAM_EXECUTOR_RAMDRIVE_SIZE="8388608" # 8GB
PROGRAM_EXECUTOR_MEMORY_SIZE="32G"

STAGE0_PATH="$ARTIFACTS_DIR/$OAK_STAGE0_DEST"
KERNEL_PATH="$ARTIFACTS_DIR/$OAK_KERNEL_DEST"
STAGE1_PATH="$ARTIFACTS_DIR/$OAK_STAGE1_DEST"
SYSTEM_IMAGE_PATH="$ARTIFACTS_DIR/$OAK_SYSTEM_IMAGE_DEST"

KMS_BUNDLE_PATH="$ARTIFACTS_DIR/kms_oci_runtime_bundle.tar"
ROOT_PROGRAM_TENSORFLOW_CONTAINER_BUNDLE="$ARTIFACTS_DIR/root_oci_runtime_bundle.tar"

CIFAR_TRAIN_FILE="/home/r14922032/cfc_tensorflow/containers/demo_cluster/testdata/cifar-10-batches-bin/data_batch_1.bin"
CIFAR_TEST_FILE="/home/r14922032/cfc_tensorflow/containers/demo_cluster/testdata/cifar-10-batches-bin/test_batch.bin"
CIFAR_MODEL_FILE="$ARTIFACTS_DIR/cifar10_model.zip"

sudo RUST_LOG=debug GLOG_v=3 "$LAUNCHER_EXEC_PATH" \
    --vmm-binary="$QEMU_PATH" \
    --stage0-binary="$STAGE0_PATH" \
    --kernel="$KERNEL_PATH" \
    --initrd="$STAGE1_PATH" \
    --system-image="$SYSTEM_IMAGE_PATH" \
    --container-bundle="$ROOT_PROGRAM_TENSORFLOW_CONTAINER_BUNDLE" \
    --ramdrive-size="$PROGRAM_EXECUTOR_RAMDRIVE_SIZE" \
    --memory-size="$PROGRAM_EXECUTOR_MEMORY_SIZE" \
    --vm-type="sev-snp" \
    --kms-bundle="$KMS_BUNDLE_PATH" \
    --kms-ramdrive-size="$KMS_RAMDRIVE_SIZE" \
    --kms-memory-size="$KMS_MEMORY_SIZE" \
    --test-type="cifar-training" \
    --cifar-data-file="$CIFAR_TRAIN_FILE" \
    --cifar-test-file="$CIFAR_TEST_FILE" \
    --num-cifar-clients=4 \
    --images-per-client=2500 \
    --num-rounds=4 \
    --local-epochs=78 \
    --learning-rate=0.02 
