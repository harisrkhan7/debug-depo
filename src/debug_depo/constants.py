"""Project defaults for the Klear AgentForge SWE-bench Verified reproduction."""

from __future__ import annotations

DEFAULT_SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SWEBENCH_SPLIT = "test"
DEFAULT_SWEBENCH_DATASET_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"
DEFAULT_SWESMITH_DATASET = "SWE-bench/SWE-smith-py"
DEFAULT_SWESMITH_SPLIT = "train"
DEFAULT_SWESMITH_DATASET_REVISION = "77cab9055d42ab4a5c25c89a8f937096db13558e"

# Override this if Kwai/Klear release the checkpoint under a different namespace.
DEFAULT_AGENTFORGE_MODEL = "Kwai-Klear/Klear-AgentForge-8B-SFT"
DEFAULT_AGENTFORGE_MODEL_REVISION = "0da97e45dbbd44278bd55b878170ec369d2934fb"
DEFAULT_LOCAL_SMOKE_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

# Repository-pinned comparison target for Klear-AgentForge-8B-SFT.
TARGET_VERIFIED_SCORE = 0.382
TARGET_VERIFIED_RESOLVED = 191
TARGET_VERIFIED_TOTAL = 500

# Pinned target setup: mini-swe-agent-plus, 200 steps, 64k context.
DEFAULT_MAX_STEPS = 200
DEFAULT_CONTEXT_LENGTH = 65536
DEFAULT_SWESMITH_CONTEXT_LENGTH = 32768
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
