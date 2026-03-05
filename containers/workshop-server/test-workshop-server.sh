QEMU_PATH="/tmp2/r14922032/qemu-v10.1.0/build/qemu-system-x86_64"

ARTIFACTS_DIR="/home/r14922032/confidential-federated-compute/artifacts"

OAK_STAGE0_DEST="stage0_bin"
OAK_KERNEL_DEST="oak_containers_kernel"
OAK_STAGE1_DEST="oak_containers_stage1.cpio"
OAK_SYSTEM_IMAGE_DEST="oak_containers_system_image.tar.xz"

RAMDRIVE_SIZE="1000000" # 1MB

LAUNCHER_EXEC_PATH="$ARTIFACTS_DIR/workshop_server"
STAGE0_PATH="$ARTIFACTS_DIR/$OAK_STAGE0_DEST"
KERNEL_PATH="$ARTIFACTS_DIR/$OAK_KERNEL_DEST"
STAGE1_PATH="$ARTIFACTS_DIR/$OAK_STAGE1_DEST"
SYSTEM_IMAGE_PATH="$ARTIFACTS_DIR/$OAK_SYSTEM_IMAGE_DEST"

KMS_BUNDLE_PATH="$ARTIFACTS_DIR/kms_oci_runtime_bundle.tar"
TEST_CONCAT_BUNDLE_PATH="$ARTIFACTS_DIR/test_concat_oci_runtime_bundle.tar"

sudo RUST_LOG=debug GLOG_v=3 "$LAUNCHER_EXEC_PATH" \
    --vmm-binary="$QEMU_PATH" \
    --stage0-binary="$STAGE0_PATH" \
    --kernel="$KERNEL_PATH" \
    --initrd="$STAGE1_PATH" \
    --system-image="$SYSTEM_IMAGE_PATH" \
    --ramdrive-size="$RAMDRIVE_SIZE" \
    --vm-type="sev-snp" \
    --kms-bundle="$KMS_BUNDLE_PATH" \
    --test-concat-bundle="$TEST_CONCAT_BUNDLE_PATH" \
    --host="0.0.0.0" \
    --port="8080" \
    --quiet
