QEMU_PATH="/tmp2/r14922032/qemu-v10.1.0/build/qemu-system-x86_64"

ARTIFACTS_DIR="/tmp2/pingchungchang/artifacts"

OAK_STAGE0_DEST="stage0_bin"
OAK_KERNEL_DEST="oak_containers_kernel"
OAK_STAGE1_DEST="oak_containers_stage1.cpio"
OAK_SYSTEM_IMAGE_DEST="oak_containers_system_image.tar.xz"

# RAMDRIVE_SIZE="1000000" # 1MB
RAMDRIVE_SIZE="1000000" # 8MB

LAUNCHER_EXEC_PATH="$ARTIFACTS_DIR/storage_proxy_launcher"
STAGE0_PATH="$ARTIFACTS_DIR/$OAK_STAGE0_DEST"
KERNEL_PATH="$ARTIFACTS_DIR/$OAK_KERNEL_DEST"
STAGE1_PATH="$ARTIFACTS_DIR/$OAK_STAGE1_DEST"
SYSTEM_IMAGE_PATH="$ARTIFACTS_DIR/$OAK_SYSTEM_IMAGE_DEST"

BUNDLE_PATH="$1"

sudo RUST_LOG=debug GLOG_v=3 "$LAUNCHER_EXEC_PATH" \
    --system_image "$SYSTEM_IMAGE_PATH" \
    --container_bundle "$BUNDLE_PATH" \
    --vmm-binary "$QEMU_PATH" \
    --stage0-binary "$STAGE0_PATH" \
    --kernel "$KERNEL_PATH" \
    --initrd "$STAGE1_PATH" \
    --ramdrive-size "$RAMDRIVE_SIZE" \
    --vm-type "sev-snp" \
