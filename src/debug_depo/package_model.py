"""Merge a trained LoRA adapter into a standalone Hugging Face model."""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path

from debug_depo.utils import requires_mistral_regex_fix


def package_model(
    base_model: str,
    adapter_path: str | Path,
    output_dir: str | Path,
    *,
    base_model_revision: str | None = None,
    max_shard_size: str = "5GB",
    trust_remote_code: bool = False,
) -> dict[str, object]:
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    adapter = Path(adapter_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if (output / "package_manifest.json").is_file() and (output / "config.json").is_file():
        return json.loads((output / "package_manifest.json").read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        preserved = output.with_name(f".{output.name}.incomplete-{uuid.uuid4().hex}")
        os.replace(output, preserved)
    staging = output.with_name(f".{output.name}.staging-{uuid.uuid4().hex}")
    staging.mkdir()
    revision_kwargs = {"revision": base_model_revision} if base_model_revision else {}
    model_config = AutoConfig.from_pretrained(
        base_model,
        trust_remote_code=trust_remote_code,
        **revision_kwargs,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        config=model_config,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
        **revision_kwargs,
    )
    model = PeftModel.from_pretrained(model, adapter)
    merged = model.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(
        staging,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer_source = adapter if (adapter / "tokenizer_config.json").is_file() else base_model
    tokenizer_revision_kwargs = revision_kwargs if tokenizer_source == base_model else {}
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
        fix_mistral_regex=requires_mistral_regex_fix(model_config),
        **tokenizer_revision_kwargs,
    )
    tokenizer.save_pretrained(staging)
    manifest = {
        "schema_version": 1,
        "format": "standalone_huggingface_model",
        "base_model": base_model,
        "base_model_revision": base_model_revision,
        "adapter_path": str(adapter),
        "output_dir": str(output),
    }
    (staging / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(staging, output)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-model-revision")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = package_model(
        args.base_model,
        args.adapter_path,
        args.output_dir,
        base_model_revision=args.base_model_revision,
        max_shard_size=args.max_shard_size,
        trust_remote_code=args.trust_remote_code,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
