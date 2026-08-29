#!/usr/bin/env bash
set -euo pipefail

CLOUD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$CLOUD_DIR/common.sh"

require_command findmnt
require_separate_storage

persistent_source="$(findmnt -n -o SOURCE -T "$CLOUD_PERSISTENT_ROOT")"
persistent_type="$(findmnt -n -o FSTYPE -T "$CLOUD_PERSISTENT_ROOT")"
local_source="$(findmnt -n -o SOURCE -T "$CLOUD_EPHEMERAL_ROOT")"
local_type="$(findmnt -n -o FSTYPE -T "$CLOUD_EPHEMERAL_ROOT")"

cat <<MSG
Cloud storage check passed.
  durable artifacts:  $CLOUD_PERSISTENT_ROOT
  durable filesystem: $persistent_source ($persistent_type)
  local rebuildables: $CLOUD_EPHEMERAL_ROOT
  local filesystem:   $local_source ($local_type)
MSG
