"""Prompt ownership note.

The AgentForge SWE-bench prompt/scaffold lives in the external Kwai/Klear harness.
This package intentionally does not duplicate it; the rollout wrapper passes each
SWE-bench instance to that harness and records the resulting patch.
"""
