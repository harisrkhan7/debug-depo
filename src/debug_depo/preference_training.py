"""Train LoRA adapters with DMPO or DEPO over multi-turn agent trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from debug_depo.utils import requires_mistral_regex_fix, write_json, write_jsonl


def dmpo_turn_weights(turns: int, gamma: float) -> list[float]:
    """Return the DMPO authors' official coefficient for each agent turn."""

    if turns < 1:
        raise ValueError("turns must be positive")
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    if gamma == 1:
        return [(turns - turn) / turns for turn in range(turns)]
    denominator = 1 - gamma**turns
    return [
        gamma**turn * (1 - gamma ** (turns - turn)) / denominator
        for turn in range(turns)
    ]


def depo_efficiency_bonus(
    row: dict[str, Any],
    *,
    alpha_tokens: float,
    alpha_steps: float,
    token_metric: str,
) -> float:
    """Calculate equation (10)'s bonus, which is zero for undesirable rows."""

    if row.get("label") != "desirable":
        return 0.0
    if token_metric not in {"completion_tokens", "total_tokens"}:
        raise ValueError("token_metric must be completion_tokens or total_tokens")
    efficiency = row.get("efficiency")
    if not isinstance(efficiency, dict):
        raise ValueError("DEPO row is missing its efficiency object")
    token_key = f"inverse_{token_metric}_per_step"
    try:
        inverse_tokens = float(efficiency[token_key])
        inverse_steps = float(efficiency["inverse_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"DEPO row has invalid {token_key}/inverse_steps values") from exc
    return alpha_tokens * inverse_tokens + alpha_steps * inverse_steps


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    if not rows:
        raise ValueError(f"No training rows found in {path}")
    return rows


def validate_training_rows(objective: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Validate builder output before any model or GPU memory is allocated."""

    required = (
        {"id", "prompt", "chosen", "rejected"}
        if objective == "dmpo"
        else {"id", "label", "prompt", "completion", "efficiency"}
    )
    labels: dict[str, int] = {}
    assistant_turns: list[int] = []
    for index, row in enumerate(rows):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Row {index} is missing: {', '.join(missing)}")
        prompt = row["prompt"]
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Row {index} has an invalid prompt")
        branches = (
            (row["chosen"], row["rejected"])
            if objective == "dmpo"
            else (row["completion"],)
        )
        for branch in branches:
            if not isinstance(branch, list) or not branch:
                raise ValueError(f"Row {index} has an empty trajectory")
            turns = sum(
                isinstance(message, dict) and message.get("role") == "assistant"
                for message in branch
            )
            if turns < 1:
                raise ValueError(f"Row {index} has no assistant turn")
            assistant_turns.append(turns)
        if objective == "depo":
            label = str(row["label"])
            if label not in {"desirable", "undesirable"}:
                raise ValueError(f"Row {index} has invalid DEPO label: {label}")
            labels[label] = labels.get(label, 0) + 1
    if objective == "depo" and set(labels) != {"desirable", "undesirable"}:
        raise ValueError("DEPO training requires both desirable and undesirable rows")
    return {
        "objective": objective,
        "rows": len(rows),
        "labels": labels,
        "min_assistant_turns": min(assistant_turns),
        "max_assistant_turns": max(assistant_turns),
    }


def _render(tokenizer: Any, messages: Sequence[dict[str, str]], *, generation: bool) -> list[int]:
    return list(
        tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=generation,
        )
    )


def tokenize_trajectory(
    tokenizer: Any,
    prompt: Sequence[dict[str, str]],
    completion: Sequence[dict[str, str]],
    *,
    max_length: int,
    turn_weights: Sequence[float] | None = None,
) -> dict[str, list[int] | list[float]]:
    """Render a chat and mark only assistant action tokens for likelihood training."""

    messages = [dict(message) for message in (*prompt, *completion)]
    full_ids = _render(tokenizer, messages, generation=False)
    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    weights = [0.0] * len(full_ids)
    assistant_indices = [
        index
        for index in range(len(prompt), len(messages))
        if messages[index].get("role") == "assistant"
    ]
    coefficients = list(turn_weights or [1.0] * len(assistant_indices))
    if len(coefficients) != len(assistant_indices):
        raise ValueError("turn_weights must contain one coefficient per assistant turn")

    for coefficient, message_index in zip(coefficients, assistant_indices):
        before = _render(tokenizer, messages[:message_index], generation=True)
        after = _render(tokenizer, messages[: message_index + 1], generation=False)
        if full_ids[: min(len(before), len(full_ids))] != before[: len(full_ids)]:
            raise ValueError(
                "The tokenizer chat template is not prefix-stable at an assistant turn"
            )
        start = min(len(before), len(full_ids))
        end = min(len(after), len(full_ids))
        for token_index in range(start, end):
            weights[token_index] = float(coefficient)

    if not any(weight > 0 for weight in weights):
        raise ValueError(
            "No assistant tokens remain after tokenization; increase --max-length"
        )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "token_weights": weights,
    }


class PreferenceDataset:
    """Lazy tokenizer-backed dataset for either preference objective."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        tokenizer: Any,
        *,
        objective: str,
        max_length: int,
        gamma: float,
        alpha_tokens: float,
        alpha_steps: float,
        token_metric: str,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.objective = objective
        self.max_length = max_length
        self.gamma = gamma
        self.alpha_tokens = alpha_tokens
        self.alpha_steps = alpha_steps
        self.token_metric = token_metric

    def __len__(self) -> int:
        return len(self.rows)

    def _trajectory(
        self,
        prompt: Sequence[dict[str, str]],
        completion: Sequence[dict[str, str]],
        *,
        dmpo: bool,
    ) -> dict[str, Any]:
        turns = sum(message.get("role") == "assistant" for message in completion)
        coefficients = dmpo_turn_weights(turns, self.gamma) if dmpo else None
        return tokenize_trajectory(
            self.tokenizer,
            prompt,
            completion,
            max_length=self.max_length,
            turn_weights=coefficients,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        if self.objective == "dmpo":
            return {
                "chosen": self._trajectory(row["prompt"], row["chosen"], dmpo=True),
                "rejected": self._trajectory(row["prompt"], row["rejected"], dmpo=True),
            }
        item = self._trajectory(row["prompt"], row["completion"], dmpo=False)
        item["desirable"] = row["label"] == "desirable"
        item["efficiency_bonus"] = depo_efficiency_bonus(
            row,
            alpha_tokens=self.alpha_tokens,
            alpha_steps=self.alpha_steps,
            token_metric=self.token_metric,
        )
        # KTO estimates KL on a prompt paired with another sample's completion.
        # Keep this deterministic so resumed/distributed jobs see identical data.
        next_row = self.rows[(index + 1) % len(self.rows)]
        item["kl"] = self._trajectory(
            row["prompt"],
            next_row["completion"],
            dmpo=False,
        )
        return item


class PreferenceCollator:
    def __init__(self, pad_token_id: int, objective: str) -> None:
        self.pad_token_id = pad_token_id
        self.objective = objective

    def _pad(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        longest = max(len(item["input_ids"]) for item in items)
        input_ids = []
        attention_mask = []
        token_weights = []
        for item in items:
            padding = longest - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            token_weights.append(item["token_weights"] + [0.0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "token_weights": torch.tensor(token_weights, dtype=torch.float32),
        }

    def __call__(self, items: Sequence[dict[str, Any]]) -> dict[str, Any]:
        import torch

        if self.objective == "dmpo":
            chosen = self._pad([item["chosen"] for item in items])
            rejected = self._pad([item["rejected"] for item in items])
            longest = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
            for branch in (chosen, rejected):
                padding = longest - branch["input_ids"].shape[1]
                if padding:
                    branch["input_ids"] = torch.nn.functional.pad(
                        branch["input_ids"], (0, padding), value=self.pad_token_id
                    )
                    branch["attention_mask"] = torch.nn.functional.pad(
                        branch["attention_mask"], (0, padding), value=0
                    )
                    branch["token_weights"] = torch.nn.functional.pad(
                        branch["token_weights"], (0, padding), value=0
                    )
            return {"chosen": chosen, "rejected": rejected}
        batch = self._pad(items)
        batch["desirable"] = torch.tensor(
            [bool(item["desirable"]) for item in items], dtype=torch.bool
        )
        batch["efficiency_bonus"] = torch.tensor(
            [float(item["efficiency_bonus"]) for item in items], dtype=torch.float32
        )
        batch["kl"] = self._pad([item["kl"] for item in items])
        return batch


class EpochShuffleSampler:
    """Use an epoch-derived permutation so a resumed epoch has the same order."""

    def __init__(self, size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(indices)
        yield from indices

    def __len__(self) -> int:
        return self.size


def _sequence_logps(model: Any, batch: dict[str, Any]) -> Any:
    import torch

    logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    ).logits
    labels = batch["input_ids"][:, 1:]
    logps = torch.gather(
        logits[:, :-1].float().log_softmax(dim=-1),
        dim=2,
        index=labels.unsqueeze(2),
    ).squeeze(2)
    return (logps * batch["token_weights"][:, 1:]).sum(dim=-1)


def _disable_dropout(model: Any) -> None:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0


@dataclass
class TrainConfig:
    objective: str
    model_name_or_path: str
    model_revision: str | None
    data_path: str
    output_dir: str
    max_rows: int
    max_length: int
    epochs: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    beta: float
    gamma: float
    desirable_weight: float
    undesirable_weight: float
    alpha_tokens: float
    alpha_steps: float
    token_metric: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: str
    gradient_checkpointing: bool
    mixed_precision: str
    seed: int
    dataloader_workers: int
    logging_steps: int
    save_steps: int
    expected_num_processes: int
    resume: bool
    trust_remote_code: bool
    attn_implementation: str | None


def _latest_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
            state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
            if int(state["global_step"]) != step:
                continue
            checkpoints.append((step, path))
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return max(checkpoints, default=(0, None))[1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _prepare_training_metrics(path: Path, resume_step: int | None) -> None:
    """Create a clean metrics log, retaining only committed resume steps."""

    retained: list[dict[str, Any]] = []
    if resume_step is not None and path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    metric = json.loads(line)
                    step = int(metric["global_step"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    # A killed process can leave one incomplete final line. The
                    # preceding complete records remain useful and recoverable.
                    break
                if step <= resume_step:
                    retained.append(metric)
    write_jsonl(path, retained)


def _append_training_metric(path: Path, metric: dict[str, Any]) -> None:
    """Durably append one loss observation to a JSONL training log."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _trial_config(config: TrainConfig) -> dict[str, Any]:
    values = asdict(config)
    values.pop("resume")
    values["data_path"] = str(Path(config.data_path).expanduser().resolve())
    values["output_dir"] = str(Path(config.output_dir).expanduser().resolve())
    requested_arm = os.getenv("EXPERIMENT_ARM", "dmpo-depo")
    experiment_arm = "dmpo" if config.objective == "dmpo" else requested_arm
    return {
        "schema_version": 1,
        # DMPO is a reusable parent artifact shared by the dmpo and dmpo-depo
        # comparison arms. Only DEPO needs its parent lineage encoded here.
        "experiment_arm": experiment_arm,
        "dmpo_trial_name": (
            os.getenv("DMPO_TRIAL_NAME", "default")
            if experiment_arm in {"dmpo", "dmpo-depo"}
            else None
        ),
        "depo_trial_name": (
            os.getenv("DEPO_TRIAL_NAME", "default")
            if config.objective == "depo"
            else None
        ),
        "data_sha256": hashlib.sha256(Path(config.data_path).read_bytes()).hexdigest(),
        "config": values,
    }


def _ensure_compatible_trial_config(output_dir: Path, payload: dict[str, Any]) -> None:
    path = output_dir / "trial_config.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            changed = sorted(
                key
                for key in set(existing.get("config", {})) | set(payload["config"])
                if existing.get("config", {}).get(key) != payload["config"].get(key)
            )
            if existing.get("data_sha256") != payload["data_sha256"]:
                changed.append("data_sha256")
            raise ValueError(
                f"Trial directory contains a different configuration: {output_dir}. "
                f"Changed fields: {', '.join(changed) or 'trial names'}. "
                "Choose a new DMPO_TRIAL_NAME/DEPO_TRIAL_NAME."
            )
        return
    if any(output_dir.iterdir()):
        raise ValueError(
            f"Trial directory is non-empty but has no trial_config.json: {output_dir}. "
            "Choose a new trial name or migrate the existing run."
        )
    _write_json(path, payload)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def train(config: TrainConfig) -> dict[str, Any]:
    """Run distributed LoRA preference training through Accelerate."""

    import torch
    import torch.nn.functional as functional
    from accelerate import Accelerator
    from accelerate.utils import broadcast_object_list, set_seed
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_jsonl(config.data_path)
    if config.max_rows:
        if config.max_rows < 1:
            raise ValueError("max_rows cannot be negative")
        indices = list(range(len(rows)))
        random.Random(config.seed).shuffle(indices)
        rows = [rows[index] for index in sorted(indices[: config.max_rows])]
    data_summary = validate_training_rows(config.objective, rows)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=None if config.mixed_precision == "no" else config.mixed_precision,
    )
    if accelerator.num_processes != config.expected_num_processes:
        raise ValueError(
            "Accelerate process count does not match the requested trial: "
            f"{accelerator.num_processes} != {config.expected_num_processes}"
        )
    config_error: str | None = None
    if accelerator.is_main_process:
        try:
            _ensure_compatible_trial_config(output_dir, _trial_config(config))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            config_error = str(exc)
    config_errors = [config_error]
    broadcast_object_list(config_errors)
    if config_errors[0]:
        raise ValueError(config_errors[0])
    accelerator.wait_for_everyone()
    complete = (
        (output_dir / "training_manifest.json").is_file()
        and (output_dir / "adapter" / "adapter_config.json").is_file()
    )
    completion_states = [complete if accelerator.is_main_process else None]
    broadcast_object_list(completion_states)
    if completion_states[0]:
        accelerator.print(f"Reusing complete {config.objective.upper()} training: {output_dir}")
        return {
            "objective": config.objective,
            "adapter_dir": str(output_dir / "adapter"),
            "global_steps": json.loads(
                (output_dir / "training_manifest.json").read_text(encoding="utf-8")
            )["global_steps"],
            "reused": True,
        }
    set_seed(config.seed)
    revision_kwargs = {"revision": config.model_revision} if config.model_revision else {}
    model_config = AutoConfig.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        **revision_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name_or_path,
        trust_remote_code=config.trust_remote_code,
        # Transformers 4.57.x can misidentify locally packaged Qwen models as
        # affected Mistral tokenizers. Apply the rewrite only to Mistral-family
        # model configurations; Qwen's existing regex is intentional.
        fix_mistral_regex=requires_mistral_regex_fix(model_config),
        **revision_kwargs,
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The selected tokenizer has no chat_template")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "dtype": (
            torch.bfloat16
            if config.mixed_precision == "bf16"
            else torch.float16 if config.mixed_precision == "fp16" else torch.float32
        ),
        "config": model_config,
        "trust_remote_code": config.trust_remote_code,
    }
    model_kwargs.update(revision_kwargs)
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    _disable_dropout(model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    target_modules = [
        value.strip() for value in config.lora_target_modules.split(",") if value.strip()
    ]
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=target_modules,
            task_type="CAUSAL_LM",
            bias="none",
        ),
    )
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    dataset = PreferenceDataset(
        rows,
        tokenizer,
        objective=config.objective,
        max_length=config.max_length,
        gamma=config.gamma,
        alpha_tokens=config.alpha_tokens,
        alpha_steps=config.alpha_steps,
        token_metric=config.token_metric,
    )
    sampler = EpochShuffleSampler(len(dataset), config.seed)
    loader = DataLoader(
        dataset,
        batch_size=config.per_device_batch_size,
        sampler=sampler,
        collate_fn=PreferenceCollator(tokenizer.pad_token_id, config.objective),
        num_workers=config.dataloader_workers,
        pin_memory=True,
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    base_optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    model, optimizer, loader = accelerator.prepare(model, base_optimizer, loader)
    updates_per_epoch = math.ceil(len(loader) / config.gradient_accumulation_steps)
    total_steps = updates_per_epoch * config.epochs
    scheduler = get_scheduler(
        config.lr_scheduler_type,
        base_optimizer,
        num_warmup_steps=round(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )
    accelerator.register_for_checkpointing(scheduler)

    start_epoch = 0
    batches_to_skip = 0
    global_step = 0
    checkpoint = _latest_checkpoint(output_dir) if config.resume else None
    if checkpoint is not None:
        accelerator.load_state(checkpoint)
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        start_epoch = int(state["epoch"])
        batches_to_skip = int(state["batch_in_epoch"])
        global_step = int(state["global_step"])
        accelerator.print(f"Resumed from {checkpoint} at update {global_step}")

    metrics_path = output_dir / "training_metrics.jsonl"
    if accelerator.is_main_process:
        _prepare_training_metrics(
            metrics_path,
            global_step if checkpoint is not None else None,
        )
    accelerator.wait_for_everyone()

    checkpoint_tokens = [uuid.uuid4().hex if accelerator.is_main_process else None]
    broadcast_object_list(checkpoint_tokens)
    checkpoint_token = str(checkpoint_tokens[0])
    last_checkpoint_step = global_step if checkpoint is not None else -1

    def save_checkpoint(epoch: int, batch_in_epoch: int) -> None:
        nonlocal last_checkpoint_step
        checkpoint_dir = output_dir / f"checkpoint-{global_step}"
        if checkpoint_dir.is_dir():
            last_checkpoint_step = global_step
            return
        temporary_dir = output_dir / f".checkpoint-{global_step}.{checkpoint_token}.tmp"
        accelerator.save_state(temporary_dir)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _write_json(
                temporary_dir / "trainer_state.json",
                {
                    "epoch": epoch,
                    "batch_in_epoch": batch_in_epoch,
                    "global_step": global_step,
                },
            )
            os.replace(temporary_dir, checkpoint_dir)
        accelerator.wait_for_everyone()
        last_checkpoint_step = global_step

    model.train()
    running_loss = 0.0
    running_loss_batches = 0
    for epoch in range(start_epoch, config.epochs):
        sampler.set_epoch(epoch)
        epoch_loader = (
            accelerator.skip_first_batches(loader, batches_to_skip)
            if epoch == start_epoch and batches_to_skip
            else loader
        )
        processed_batches = batches_to_skip if epoch == start_epoch else 0
        for batch in epoch_loader:
            processed_batches += 1
            with accelerator.accumulate(model):
                if config.objective == "dmpo":
                    combined = {
                        key: torch.cat((batch["chosen"][key], batch["rejected"][key]), dim=0)
                        for key in ("input_ids", "attention_mask", "token_weights")
                    }
                    policy_logps = _sequence_logps(model, combined)
                    reference_context = accelerator.unwrap_model(model).disable_adapter()
                    with torch.no_grad(), reference_context:
                        reference_logps = _sequence_logps(model, combined)
                    midpoint = policy_logps.shape[0] // 2
                    policy_margin = policy_logps[:midpoint] - policy_logps[midpoint:]
                    reference_margin = (
                        reference_logps[:midpoint] - reference_logps[midpoint:]
                    )
                    loss = -functional.logsigmoid(
                        config.beta * (policy_margin - reference_margin)
                    ).mean()
                else:
                    policy_logps = _sequence_logps(model, batch)
                    with torch.no_grad():
                        policy_kl_logps = _sequence_logps(model, batch["kl"])
                    reference_context = accelerator.unwrap_model(model).disable_adapter()
                    with torch.no_grad(), reference_context:
                        reference_logps = _sequence_logps(model, batch)
                        reference_kl_logps = _sequence_logps(model, batch["kl"])
                    logratios = policy_logps - reference_logps
                    kl_logratios = policy_kl_logps - reference_kl_logps
                    gathered_kl = accelerator.gather(kl_logratios)
                    kl = gathered_kl.mean().clamp(min=0)
                    desirable = batch["desirable"]
                    bonus = batch["efficiency_bonus"].to(logratios.dtype)
                    desirable_values = 1 - torch.sigmoid(
                        config.beta * (logratios - kl + bonus)
                    )
                    undesirable_values = 1 - torch.sigmoid(
                        config.beta * (kl - logratios)
                    )
                    losses = torch.where(
                        desirable,
                        config.desirable_weight * desirable_values,
                        config.undesirable_weight * undesirable_values,
                    )
                    loss = losses.mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            mean_loss = accelerator.gather(loss.detach().float().reshape(1)).mean().item()
            running_loss += mean_loss
            running_loss_batches += 1
            if accelerator.sync_gradients:
                global_step += 1
                if global_step % config.logging_steps == 0:
                    logged_loss = running_loss / running_loss_batches
                    accelerator.print(
                        f"step={global_step}/{total_steps} "
                        f"loss={logged_loss:.6f}"
                    )
                    if accelerator.is_main_process:
                        _append_training_metric(
                            metrics_path,
                            {
                                "epoch": epoch + 1,
                                "global_step": global_step,
                                "learning_rate": scheduler.get_last_lr()[0],
                                "loss": logged_loss,
                                "micro_batches": running_loss_batches,
                            },
                        )
                    running_loss = 0.0
                    running_loss_batches = 0
                if config.save_steps and global_step % config.save_steps == 0:
                    save_checkpoint(epoch, processed_batches)
        if global_step != last_checkpoint_step:
            save_checkpoint(epoch + 1, 0)
        batches_to_skip = 0

    if running_loss_batches:
        logged_loss = running_loss / running_loss_batches
        accelerator.print(
            f"step={global_step}/{total_steps} loss={logged_loss:.6f} (final)"
        )
        if accelerator.is_main_process:
            _append_training_metric(
                metrics_path,
                {
                    "epoch": config.epochs,
                    "global_step": global_step,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "loss": logged_loss,
                    "micro_batches": running_loss_batches,
                },
            )

    accelerator.wait_for_everyone()
    adapter_dir = output_dir / "adapter"
    unwrapped = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        unwrapped.save_pretrained(
            adapter_dir,
            is_main_process=True,
            save_function=accelerator.save,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(adapter_dir)
        manifest = {
            "schema_version": 1,
            "objective": config.objective,
            "base_and_reference_model": config.model_name_or_path,
            "base_and_reference_model_revision": config.model_revision,
            "adapter_dir": str(adapter_dir),
            "data_path": str(Path(config.data_path).resolve()),
            "data_sha256": hashlib.sha256(Path(config.data_path).read_bytes()).hexdigest(),
            "data_summary": data_summary,
            "git_commit": _git_commit(),
            "global_steps": global_step,
            "metrics_path": str(metrics_path),
            "config": asdict(config),
        }
        _write_json(output_dir / "training_manifest.json", manifest)
    accelerator.wait_for_everyone()
    return {
        "objective": config.objective,
        "adapter_dir": str(adapter_dir),
        "global_steps": global_step,
        "reused": False,
    }


def build_parser(objective: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    if objective is None:
        parser.add_argument("--objective", choices=("dmpo", "depo"), required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--lr-scheduler-type", default="constant_with_warmup")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--desirable-weight", type=float, default=1.0)
    parser.add_argument("--undesirable-weight", type=float, default=1.0)
    parser.add_argument("--alpha-tokens", type=float, default=2.0)
    parser.add_argument("--alpha-steps", type=float, default=2.0)
    parser.add_argument(
        "--token-metric",
        choices=("completion_tokens", "total_tokens"),
        default="total_tokens",
    )
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader-workers", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--expected-num-processes", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None, *, objective: str | None = None) -> int:
    args = build_parser(objective).parse_args(argv)
    selected_objective = objective or args.objective
    rows = _read_jsonl(args.data_path)
    summary = validate_training_rows(selected_objective, rows)
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return 0
    if args.max_length < 2:
        raise ValueError("max_length must be at least 2")
    config = TrainConfig(
        objective=selected_objective,
        model_name_or_path=args.model_name_or_path,
        model_revision=args.model_revision,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        max_length=args.max_length,
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        beta=args.beta,
        gamma=args.gamma,
        desirable_weight=args.desirable_weight,
        undesirable_weight=args.undesirable_weight,
        alpha_tokens=args.alpha_tokens,
        alpha_steps=args.alpha_steps,
        token_metric=args.token_metric,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        dataloader_workers=args.dataloader_workers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        expected_num_processes=args.expected_num_processes,
        resume=not args.no_resume,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation or None,
    )
    result = train(config)
    print(json.dumps(result, indent=2))
    return 0


def dmpo_main(argv: list[str] | None = None) -> int:
    return main(argv, objective="dmpo")


def depo_main(argv: list[str] | None = None) -> int:
    return main(argv, objective="depo")


if __name__ == "__main__":
    raise SystemExit(main())
