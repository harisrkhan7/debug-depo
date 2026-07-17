#!/usr/bin/env bash
set -euo pipefail

# Store a Hugging Face read token outside the repo for vLLM / Apptainer runs.
# The token is not printed and the file is readable only by the current user.

HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.config/debug-depo/hf_token}"
token_dir="$(dirname "$HF_TOKEN_FILE")"

mkdir -p "$token_dir"
chmod 700 "$token_dir"

if [[ -n "${HF_TOKEN:-}" ]]; then
  token="$HF_TOKEN"
else
  read -r -s -p "Paste Hugging Face token: " token
  echo
fi

if [[ -z "$token" ]]; then
  echo "No token provided." >&2
  exit 1
fi

if [[ "$token" != hf_* ]]; then
  echo "Warning: Hugging Face tokens usually start with 'hf_'." >&2
fi

umask 077
printf "%s" "$token" > "$HF_TOKEN_FILE"
chmod 600 "$HF_TOKEN_FILE"
unset token

cat <<MSG
Saved Hugging Face token to:
  $HF_TOKEN_FILE

Permissions:
  $(ls -l "$HF_TOKEN_FILE")

The token is outside the repo and will be loaded by the vLLM and rollout helpers.
MSG
