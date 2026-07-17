"""Project defaults for the Klear AgentForge SWE-bench Verified reproduction."""

from __future__ import annotations

DEFAULT_SWEBENCH_DATASET = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SWEBENCH_SPLIT = "test"

# Override this if Kwai/Klear release the checkpoint under a different namespace.
DEFAULT_AGENTFORGE_MODEL = "Kwai-Klear/Klear-AgentForge-8B-SFT"
DEFAULT_LOCAL_SMOKE_MODEL = "mlx-community/Qwen2.5-Coder-7B-Instruct-4bit"

# The paper reports Klear-AgentForge-8B-SFT at 38.2 on SWE-bench Verified.
TARGET_VERIFIED_SCORE = 0.382
TARGET_VERIFIED_RESOLVED = 191
TARGET_VERIFIED_TOTAL = 500

# Paper setup: mini-swe-agent-plus, 200 steps, 64k context.
DEFAULT_MAX_STEPS = 200
DEFAULT_CONTEXT_LENGTH = 65536
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
