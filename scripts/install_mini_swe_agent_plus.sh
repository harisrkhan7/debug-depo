#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="${UV:-uv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

DEST="${MINI_SWE_AGENT_PLUS_REPO:-$ROOT_DIR/external/mini-swe-agent-plus}"
REPO_URL="${MINI_SWE_AGENT_PLUS_URL:-https://github.com/Kwai-Klear/mini-swe-agent-plus.git}"

cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  "$UV_BIN" sync --extra dev
fi

PROJECT_PYTHON="$("$UV_BIN" run python -c 'import sys; print(sys.executable)')"

if [[ ! -d "$DEST/.git" ]]; then
  mkdir -p "$(dirname "$DEST")"
  git clone "$REPO_URL" "$DEST"
else
  git -C "$DEST" pull --ff-only
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

old = (
    "                subprocess.run(\n"
    "                    [self.config.executable, \"build\", \"--sandbox\", sandbox_dir, self.config.image],\n"
    "                    check=True,\n"
    "                    capture_output=True,\n"
    "                )\n"
    "                break\n"
)
new = (
    "                subprocess.run(\n"
    "                    [self.config.executable, \"build\", \"--sandbox\", sandbox_dir, self.config.image],\n"
    "                    check=True,\n"
    "                    capture_output=True,\n"
    "                )\n"
    "                self._prepare_writable_bind_destinations(sandbox_dir)\n"
    "                break\n"
)
if old in text:
    text = text.replace(old, new, 1)
elif new not in text:
    raise SystemExit(f"Could not patch Apptainer writable bind setup in {path}")

method = (
    "    def _prepare_writable_bind_destinations(self, sandbox_dir: Path) -> None:\n"
    "        destinations = os.getenv(\"MSWEA_SINGULARITY_WRITABLE_BIND_DESTINATIONS\", \"/rds\")\n"
    "        for destination in destinations.split(os.pathsep):\n"
    "            destination = destination.strip()\n"
    "            if not destination or not destination.startswith(\"/\"):\n"
    "                continue\n"
    "            (sandbox_dir / destination.lstrip(\"/\")).mkdir(parents=True, exist_ok=True)\n"
    "\n"
)
if method not in text:
    marker = "    def get_template_vars(self) -> dict[str, Any]:\n"
    if marker not in text:
        raise SystemExit(f"Could not insert Apptainer writable bind helper in {path}")
    text = text.replace(marker, method + marker, 1)

path.write_text(text)
PY
fi

"$UV_BIN" pip install --python "$PROJECT_PYTHON" -e "$DEST"

"$UV_BIN" run python - <<'PY'
from minisweagent.config import builtin_config_dir
print(f"mini-swe-agent-plus config dir: {builtin_config_dir}")
PY

cat <<MSG
Installed official mini-swe-agent-plus from:
  $DEST
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
