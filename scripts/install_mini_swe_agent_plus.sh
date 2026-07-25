#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DEST="${MINI_SWE_AGENT_PLUS_REPO:-$ROOT_DIR/external/mini-swe-agent-plus}"
REPO_URL="${MINI_SWE_AGENT_PLUS_URL:-https://github.com/Kwai-Klear/mini-swe-agent-plus.git}"
REVISION="${MINI_SWE_AGENT_PLUS_REVISION:-3dfa5e26831306978ff3cfa2da15b49113ded0e6}"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  "$UV_BIN" sync --extra dev
fi

PROJECT_PYTHON="$("$UV_BIN" run python -c 'import sys; print(sys.executable)')"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$REPO_URL" "$DEST"
fi

if ! git -C "$DEST" cat-file -e "$REVISION^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch origin "$REVISION"
  REVISION=FETCH_HEAD
fi
TARGET_REVISION="$(git -C "$DEST" rev-parse "$REVISION^{commit}")"
CURRENT_REVISION="$(git -C "$DEST" rev-parse HEAD)"
if [[ "$CURRENT_REVISION" != "$TARGET_REVISION" ]]; then
  if ! git -C "$DEST" diff --quiet || ! git -C "$DEST" diff --cached --quiet; then
    echo "Cannot switch mini-swe-agent-plus revisions with local changes in $DEST." >&2
    echo "Clean or recreate that ignored external checkout, then rerun setup." >&2
    exit 1
  fi
  git -C "$DEST" checkout --detach "$TARGET_REVISION"
fi

LITELLM_MODEL="$DEST/src/minisweagent/models/litellm_model.py"
if [[ -f "$LITELLM_MODEL" ]]; then
  "$UV_BIN" run python - "$LITELLM_MODEL" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

hash_import = "import hashlib\n"
if hash_import not in text:
    text = text.replace("import json\n", "import json\nimport hashlib\n", 1)

hash_func = (
    "def str_hash_to_int(value: str) -> int:\n"
    "    return int(hashlib.sha256(value.encode(\"utf-8\")).hexdigest(), 16)\n"
)
if hash_func not in text:
    marker = 'logger = logging.getLogger("litellm_model")\n\n\n'
    if marker not in text:
        raise SystemExit(f"Could not patch local vLLM hash bug in {path}")
    text = text.replace(marker, marker + hash_func + "\n\n", 1)

old = (
    '        if local_vllm_server_ips_filename is not None:\n'
    '            if os.path.exists(fnm):\n'
)
new = (
    '        if local_vllm_server_ips_filename is not None:\n'
    '            fnm = local_vllm_server_ips_filename\n'
    '            if os.path.exists(fnm):\n'
)
if old in text:
    text = text.replace(old, new)
elif new not in text:
    raise SystemExit(f"Could not patch local vLLM filename bug in {path}")

path.write_text(text)
PY
fi

ADD_EDIT_CONFIG="$DEST/src/minisweagent/config/extra/swebench_add_edit_tool.yaml"
if [[ -f "$ADD_EDIT_CONFIG" ]]; then
  "$UV_BIN" run python - "$ADD_EDIT_CONFIG" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "    - MODIFY: Regular source code files in {{working_dir}}\n"
new = (
    "    - MODIFY: Regular source code files in /testbed "
    "(this is the working directory for all your subsequent commands)\n"
)
if old in text:
    path.write_text(text.replace(old, new))
elif new not in text:
    raise SystemExit(f"Could not patch working_dir template bug in {path}")
PY
fi

SINGULARITY_ENV="$DEST/src/minisweagent/environments/singularity.py"
if [[ -f "$SINGULARITY_ENV" ]]; then
  "$UV_BIN" run python - "$SINGULARITY_ENV" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()

original_build = (
    "                subprocess.run(\n"
    "                    [self.config.executable, \"build\", \"--sandbox\", sandbox_dir, self.config.image],\n"
    "                    check=True,\n"
    "                    capture_output=True,\n"
    "                )\n"
    "                break\n"
)
writable_build = (
    "                subprocess.run(\n"
    "                    [self.config.executable, \"build\", \"--sandbox\", sandbox_dir, self.config.image],\n"
    "                    check=True,\n"
    "                    capture_output=True,\n"
    "                )\n"
    "                self._prepare_writable_bind_destinations(sandbox_dir)\n"
    "                break\n"
)
cached_build = (
    "        image_source = self._cached_image_source()\n"
    "        for attempt in range(max_retries):\n"
    "            sandbox_dir = Path(tempfile.gettempdir()) / f\"minisweagent-{uuid.uuid4().hex[:8]}\"\n"
    "            try:\n"
    "                subprocess.run(\n"
    "                    [self.config.executable, \"build\", \"--sandbox\", sandbox_dir, image_source],\n"
    "                    check=True,\n"
    "                    capture_output=True,\n"
    "                )\n"
    "                self._prepare_writable_bind_destinations(sandbox_dir)\n"
    "                break\n"
)
loop_marker = (
    "        for attempt in range(max_retries):\n"
    "            sandbox_dir = Path(tempfile.gettempdir()) / f\"minisweagent-{uuid.uuid4().hex[:8]}\"\n"
    "            try:\n"
)
if "        image_source = self._cached_image_source()\n" not in text:
    if writable_build in text:
        text = text.replace(loop_marker + writable_build, cached_build, 1)
    elif original_build in text:
        text = text.replace(loop_marker + original_build, cached_build, 1)
    else:
        raise SystemExit(f"Could not patch persistent Apptainer SIF caching in {path}")

cache_method = (
    "    def _cached_image_source(self) -> str:\n"
    "        sif_dir = os.getenv(\"MSWEA_SINGULARITY_SIF_DIR\")\n"
    "        if not sif_dir or not self.config.image.startswith(\n"
    "            (\"docker://\", \"oras://\", \"library://\")\n"
    "        ):\n"
    "            return self.config.image\n"
    "\n"
    "        from debug_depo.apptainer_cache import (\n"
    "            pull_sif_if_missing,\n"
    "            sif_path_for_image,\n"
    "        )\n"
    "\n"
    "        sif_path = sif_path_for_image(sif_dir, self.config.image)\n"
    "        pull_sif_if_missing(\n"
    "            sif_path=sif_path,\n"
    "            image_uri=self.config.image,\n"
    "            cache_dir=(\n"
    "                os.getenv(\"MSWEA_SINGULARITY_CACHE_DIR\")\n"
    "                or os.getenv(\"APPTAINER_CACHEDIR\")\n"
    "            ),\n"
    "            executable=self.config.executable,\n"
    "        )\n"
    "        return str(sif_path)\n"
    "\n"
)
writable_method = (
    "    def _prepare_writable_bind_destinations(self, sandbox_dir: Path) -> None:\n"
    "        destinations = os.getenv(\"MSWEA_SINGULARITY_WRITABLE_BIND_DESTINATIONS\", \"/rds\")\n"
    "        for destination in destinations.split(os.pathsep):\n"
    "            destination = destination.strip()\n"
    "            if not destination or not destination.startswith(\"/\"):\n"
    "                continue\n"
    "            (sandbox_dir / destination.lstrip(\"/\")).mkdir(parents=True, exist_ok=True)\n"
    "\n"
)
methods = cache_method + writable_method
if cache_method not in text:
    marker = "    def get_template_vars(self) -> dict[str, Any]:\n"
    if marker not in text:
        raise SystemExit(f"Could not insert Apptainer cache helpers in {path}")
    if writable_method in text:
        text = text.replace(writable_method, methods, 1)
    else:
        text = text.replace(marker, methods + marker, 1)

old_cleanup = "    def cleanup(self):\n        shutil.rmtree(self.sandbox_dir, ignore_errors=True)\n"
new_cleanup = (
    "    def cleanup(self):\n"
    "        sandbox_dir = getattr(self, \"sandbox_dir\", None)\n"
    "        if sandbox_dir is not None:\n"
    "            shutil.rmtree(sandbox_dir, ignore_errors=True)\n"
)
if old_cleanup in text:
    text = text.replace(old_cleanup, new_cleanup, 1)
elif new_cleanup not in text:
    raise SystemExit(f"Could not patch partial Singularity initialization cleanup in {path}")

path.write_text(text)
PY
fi

"$UV_BIN" pip install --python "$PROJECT_PYTHON" -e "$DEST"

"$UV_BIN" run python - <<'PY'
from minisweagent.config import builtin_config_dir
from minisweagent.environments.singularity import SingularityEnvironment

if not callable(getattr(SingularityEnvironment, "_cached_image_source", None)):
    raise SystemExit(
        "mini-swe-agent-plus installation is missing the persistent Apptainer "
        "SIF integration"
    )

print(f"mini-swe-agent-plus config dir: {builtin_config_dir}")
print("mini-swe-agent-plus persistent SIF integration: OK")
PY

cat <<MSG
Installed official mini-swe-agent-plus from:
  $DEST
Pinned revision:
  $TARGET_REVISION
Into Python:
  $PROJECT_PYTHON

Try a local no-model smoke first:
  MOCK=1 LIMIT=1 scripts/collect_rollouts.sh

For a real mini-swe-agent-plus rollout, set your OpenAI-compatible model server
and run for one task:
  HARNESS=mini-swe-agent-plus \\
  MINI_SWE_MODEL=hosted_vllm/Kwai-Klear/Klear-AgentForge-8B-SFT \\
  LLM_BASE_URL=http://127.0.0.1:8000/v1 \\
  LLM_API_KEY=local \\
  LIMIT=1 scripts/collect_rollouts.sh

On an Apptainer-only cluster, add:
  MINI_SWE_RUNNER=singularity \\
  MINI_SWE_ENVIRONMENT_CLASS=singularity \\
  MSWEA_SINGULARITY_EXECUTABLE=apptainer
MSG
