#!/usr/bin/env bash
set -euo pipefail

# Mount an already attached HyperStack Shared Storage Volume at the persistent
# root. Formatting is allowed only with the explicit FORMAT_EMPTY_DEVICE=1
# opt-in and only when blkid reports no existing filesystem.

HYPERSTACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$HYPERSTACK_DIR/local.env" ]]; then
  # shellcheck disable=SC1091
  source "$HYPERSTACK_DIR/local.env"
fi
HYPERSTACK_PERSISTENT_ROOT="${HYPERSTACK_PERSISTENT_ROOT:-/root/debug-depo-persistent}"

DEVICE="${1:-}"
MOUNT_POINT="${2:-$HYPERSTACK_PERSISTENT_ROOT}"
if [[ -z "$DEVICE" ]]; then
  echo "Usage: FORMAT_EMPTY_DEVICE=1 sudo bash hyperstack/prepare_volume.sh /dev/disk/by-id/<volume> [mount-point]" >&2
  exit 2
fi
if [[ "$(id -u)" != "0" ]]; then
  echo "Run this mount preparation script as root." >&2
  exit 2
fi
if [[ ! -b "$DEVICE" ]]; then
  echo "Not a block device: $DEVICE" >&2
  exit 2
fi

filesystem_type="$(blkid -s TYPE -o value "$DEVICE" 2>/dev/null || true)"
if [[ -z "$filesystem_type" ]]; then
  if [[ "${FORMAT_EMPTY_DEVICE:-0}" != "1" ]]; then
    echo "$DEVICE has no filesystem." >&2
    echo "Set FORMAT_EMPTY_DEVICE=1 to explicitly format this empty device as ext4." >&2
    exit 2
  fi
  mkfs.ext4 -L debug-depo-persistent "$DEVICE"
  filesystem_type=ext4
fi

if [[ -d "$MOUNT_POINT" ]] && ! mountpoint -q "$MOUNT_POINT" && \
  find "$MOUNT_POINT" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to hide existing data under non-empty mount point: $MOUNT_POINT" >&2
  echo "Move that data aside or choose an empty mount point first." >&2
  exit 2
fi
mkdir -p "$MOUNT_POINT"
if ! mountpoint -q "$MOUNT_POINT"; then
  mount "$DEVICE" "$MOUNT_POINT"
fi

device_uuid="$(blkid -s UUID -o value "$DEVICE")"
fstab_entry="UUID=$device_uuid $MOUNT_POINT $filesystem_type defaults,nofail 0 2"
if ! grep -Fq "UUID=$device_uuid " /etc/fstab; then
  printf '%s\n' "$fstab_entry" >>/etc/fstab
fi

mkdir -p "$MOUNT_POINT/scratch" "$MOUNT_POINT/tools"
echo "Persistent volume mounted at $MOUNT_POINT and recorded in /etc/fstab."
