"""Build one CSV ledger from preference-training and evaluation artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


VALIDATION_BUDGETS = (100, 200, 500)
BASE_FIELDS = (
    "trial_id",
    "objective",
    "experiment_arm",
    "dmpo_trial_name",
    "depo_trial_name",
    "parent_trial_id",
    "base_and_reference_model",
    "model_path",
    "data_path",
    "data_sha256",
    "learning_rate",
    "beta",
    "gamma",
    "token_metric",
    "alpha_tokens",
    "alpha_steps",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "expected_num_processes",
    "effective_global_batch_size",
    "epochs",
    "max_length",
    "global_steps",
    "git_commit",
    "training_complete",
    "package_complete",
    "trial_config_path",
    "training_manifest_path",
    "package_manifest_path",
)
VALIDATION_FIELD_SUFFIXES = (
    "comparison_path",
    "baseline",
    "task_matrix_sha256",
    "resolution_rate",
    "resolution_rate_delta_vs_baseline",
    "total_tokens_per_resolved_task",
    "mean_total_tokens_delta_vs_baseline",
    "mean_action_steps_delta_vs_baseline",
    "success_noninferior",
    "selection_eligible",
    "rank",
    "selected",
    "gained",
    "lost",
)
TRAILING_FIELDS = ("latest_budget", "latest_result")
FIELDNAMES = (
    *BASE_FIELDS,
    *(
        f"val_{budget}_{suffix}"
        for budget in VALIDATION_BUDGETS
        for suffix in VALIDATION_FIELD_SUFFIXES
    ),
    *TRAILING_FIELDS,
)


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {label} JSON at {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _canonical_trial_id(payload: Mapping[str, Any], *, path: Path) -> str:
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Trial config has no config object: {path}")
    objective = str(config.get("objective") or "")
    experiment_arm = str(payload.get("experiment_arm") or "")
    dmpo_name = str(payload.get("dmpo_trial_name") or "")
    depo_name = str(payload.get("depo_trial_name") or "")
    if objective == "dmpo" and dmpo_name:
        return f"dmpo/{dmpo_name}"
    if objective == "depo" and experiment_arm == "depo" and depo_name:
        return f"depo/{depo_name}"
    if objective == "depo" and experiment_arm == "dmpo-depo" and dmpo_name and depo_name:
        return f"dmpo-depo/{dmpo_name}/{depo_name}"
    raise ValueError(
        f"Cannot derive a canonical trial ID from objective={objective!r}, "
        f"experiment_arm={experiment_arm!r}, dmpo_trial_name={dmpo_name!r}, "
        f"depo_trial_name={depo_name!r}: {path}"
    )


def _optional_manifest(path: Path, *, label: str) -> dict[str, Any] | None:
    return _read_json_object(path, label=label) if path.is_file() else None


def _model_key(value: str | Path) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return os.path.normpath(str(path))
    return os.path.normpath(str(path))


def _discover_trials(run_root: Path) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    experiments_root = run_root / "experiments"
    config_paths = sorted(experiments_root.glob("**/training/trial_config.json"))
    if not config_paths:
        raise FileNotFoundError(f"No trial_config.json files found under {experiments_root}")

    trials: list[dict[str, Any]] = []
    model_index: dict[str, set[str]] = defaultdict(set)
    seen_trial_ids: set[str] = set()
    for config_path in config_paths:
        payload = _read_json_object(config_path, label="trial config")
        config = payload["config"]
        trial_id = _canonical_trial_id(payload, path=config_path)
        if trial_id in seen_trial_ids:
            raise ValueError(f"Duplicate canonical trial ID {trial_id!r}: {config_path}")
        seen_trial_ids.add(trial_id)

        training_dir = config_path.parent
        trial_root = training_dir.parent
        model_dir = trial_root / "model"
        training_manifest_path = training_dir / "training_manifest.json"
        package_manifest_path = model_dir / "package_manifest.json"
        training_manifest = _optional_manifest(
            training_manifest_path,
            label="training manifest",
        )
        package_manifest = _optional_manifest(
            package_manifest_path,
            label="package manifest",
        )

        config_data_sha = str(payload.get("data_sha256") or "")
        if training_manifest is not None:
            manifest_data_sha = str(training_manifest.get("data_sha256") or "")
            if manifest_data_sha and manifest_data_sha != config_data_sha:
                raise ValueError(
                    f"Data SHA-256 mismatch between {config_path} and "
                    f"{training_manifest_path}"
                )

        objective = str(config["objective"])
        experiment_arm = str(payload["experiment_arm"])
        dmpo_name = str(payload.get("dmpo_trial_name") or "")
        depo_name = str(payload.get("depo_trial_name") or "")
        if objective == "depo" and experiment_arm == "dmpo-depo":
            parent_trial_id = f"dmpo/{dmpo_name}"
        else:
            parent_trial_id = ""

        expected_processes = config.get("expected_num_processes")
        per_device_batch = config.get("per_device_batch_size")
        accumulation = config.get("gradient_accumulation_steps")
        effective_batch = ""
        if all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (expected_processes, per_device_batch, accumulation)
        ):
            effective_batch = expected_processes * per_device_batch * accumulation

        row: dict[str, Any] = {field: "" for field in FIELDNAMES}
        row.update(
            {
                "trial_id": trial_id,
                "objective": objective,
                "experiment_arm": experiment_arm,
                "dmpo_trial_name": dmpo_name,
                "depo_trial_name": depo_name,
                "parent_trial_id": parent_trial_id,
                "base_and_reference_model": config.get("model_name_or_path", ""),
                "model_path": _relative_path(model_dir, run_root),
                "data_path": config.get("data_path", ""),
                "data_sha256": config_data_sha,
                "learning_rate": config.get("learning_rate", ""),
                "beta": config.get("beta", ""),
                "gamma": config.get("gamma", "") if objective == "dmpo" else "",
                "token_metric": config.get("token_metric", "") if objective == "depo" else "",
                "alpha_tokens": config.get("alpha_tokens", "") if objective == "depo" else "",
                "alpha_steps": config.get("alpha_steps", "") if objective == "depo" else "",
                "per_device_batch_size": per_device_batch,
                "gradient_accumulation_steps": accumulation,
                "expected_num_processes": expected_processes,
                "effective_global_batch_size": effective_batch,
                "epochs": config.get("epochs", ""),
                "max_length": config.get("max_length", ""),
                "global_steps": (
                    training_manifest.get("global_steps", "")
                    if training_manifest is not None
                    else ""
                ),
                "git_commit": (
                    training_manifest.get("git_commit", "")
                    if training_manifest is not None
                    else ""
                ),
                "training_complete": training_manifest is not None,
                "package_complete": (
                    package_manifest is not None and (model_dir / "config.json").is_file()
                ),
                "trial_config_path": _relative_path(config_path, run_root),
                "training_manifest_path": (
                    _relative_path(training_manifest_path, run_root)
                    if training_manifest is not None
                    else ""
                ),
                "package_manifest_path": (
                    _relative_path(package_manifest_path, run_root)
                    if package_manifest is not None
                    else ""
                ),
            }
        )
        trials.append(row)

        model_values = {str(model_dir), str(model_dir.resolve())}
        if package_manifest is not None and package_manifest.get("output_dir"):
            model_values.add(str(package_manifest["output_dir"]))
        for model_value in model_values:
            model_index[_model_key(model_value)].add(trial_id)

    return sorted(trials, key=lambda row: str(row["trial_id"])), model_index


def _trial_alias_index(trials: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for trial in trials:
        trial_id = str(trial["trial_id"])
        aliases[trial_id].add(trial_id)
        leaf_name = (
            str(trial["dmpo_trial_name"])
            if trial["objective"] == "dmpo"
            else str(trial["depo_trial_name"])
        )
        if leaf_name:
            aliases[leaf_name].add(trial_id)
    return aliases


def _candidate_arm_paths(
    raw_path: str | Path,
    *,
    training_run_root: Path,
) -> list[Path]:
    path = Path(raw_path).expanduser()
    candidates = [path]
    if len(path.parents) >= 2:
        evaluation_run_name = path.parent.parent.name
        candidates.append(
            training_run_root.parent
            / evaluation_run_name
            / path.parent.name
            / path.name
        )
    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def _evaluation_model(
    arm_path: str | Path,
    *,
    training_run_root: Path,
) -> str | None:
    for candidate in _candidate_arm_paths(
        arm_path,
        training_run_root=training_run_root,
    ):
        if not candidate.is_file():
            continue
        evaluation_root = candidate.parent.parent
        manifest_paths = sorted(
            (evaluation_root / "collection").glob("shard-*/collection_manifest.json")
        )
        model_values: set[str] = set()
        for manifest_path in manifest_paths:
            manifest = _read_json_object(manifest_path, label="collection manifest")
            model = manifest.get("model")
            if model:
                model_values.add(str(model))
        if len(model_values) > 1:
            raise ValueError(
                f"Evaluation shards disagree on the model under {evaluation_root}: "
                f"{sorted(model_values)}"
            )
        if model_values:
            return next(iter(model_values))
    return None


def _unique_match(
    matches: set[str],
    *,
    description: str,
) -> str | None:
    if len(matches) > 1:
        raise ValueError(
            f"{description} matches multiple trials: {', '.join(sorted(matches))}. "
            "Use a canonical trial ID or an explicit --arm-trial mapping."
        )
    return next(iter(matches)) if matches else None


def _match_arm(
    *,
    arm_name: str,
    arm_path: str | Path,
    trials_by_id: Mapping[str, Mapping[str, Any]],
    aliases: Mapping[str, set[str]],
    model_index: Mapping[str, set[str]],
    explicit_arm_trials: Mapping[str, str],
    training_run_root: Path,
) -> str | None:
    if arm_name in explicit_arm_trials:
        trial_id = explicit_arm_trials[arm_name]
        if trial_id not in trials_by_id:
            raise ValueError(
                f"Explicit mapping for arm {arm_name!r} references unknown trial "
                f"{trial_id!r}"
            )
        return trial_id

    evaluation_model = _evaluation_model(
        arm_path,
        training_run_root=training_run_root,
    )
    if evaluation_model:
        matched = _unique_match(
            set(model_index.get(_model_key(evaluation_model), set())),
            description=f"Evaluation model {evaluation_model!r}",
        )
        if matched is not None:
            return matched

    return _unique_match(
        set(aliases.get(arm_name, set())),
        description=f"Arm name {arm_name!r}",
    )


def _nested_value(payload: Mapping[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return ""
        value = value.get(key)
    return "" if value is None else value


def _add_measurement(
    row: dict[str, Any],
    *,
    budget: int,
    comparison_path: Path,
    comparison: Mapping[str, Any],
    arm: Mapping[str, Any],
    rank: int | str,
) -> None:
    prefix = f"val_{budget}_"
    if row[f"{prefix}comparison_path"]:
        raise ValueError(
            f"Trial {row['trial_id']!r} has more than one comparison result "
            f"for budget {budget}"
        )
    efficiency = arm.get("efficiency")
    if not isinstance(efficiency, Mapping):
        raise ValueError(
            f"Comparison arm {arm.get('name')!r} has no efficiency object: "
            f"{comparison_path}"
        )
    row.update(
        {
            f"{prefix}comparison_path": str(comparison_path.resolve()),
            f"{prefix}baseline": comparison.get("baseline", ""),
            f"{prefix}task_matrix_sha256": comparison.get("task_matrix_sha256", ""),
            f"{prefix}resolution_rate": efficiency.get("resolution_rate", ""),
            f"{prefix}resolution_rate_delta_vs_baseline": arm.get(
                "resolution_rate_delta_vs_baseline",
                "",
            ),
            f"{prefix}total_tokens_per_resolved_task": efficiency.get(
                "total_tokens_per_resolved_task",
                "",
            ),
            f"{prefix}mean_total_tokens_delta_vs_baseline": _nested_value(
                arm,
                "paired_deltas_vs_baseline",
                "all",
                "total_tokens",
                "mean",
            ),
            f"{prefix}mean_action_steps_delta_vs_baseline": _nested_value(
                arm,
                "paired_deltas_vs_baseline",
                "all",
                "action_steps",
                "mean",
            ),
            f"{prefix}success_noninferior": arm.get("success_noninferior", ""),
            f"{prefix}selection_eligible": arm.get("selection_eligible", ""),
            f"{prefix}rank": rank,
            f"{prefix}selected": arm.get("name") == comparison.get("selected_arm"),
            f"{prefix}gained": _nested_value(
                arm,
                "resolution_transitions_vs_baseline",
                "gained",
            ),
            f"{prefix}lost": _nested_value(
                arm,
                "resolution_transitions_vs_baseline",
                "lost",
            ),
        }
    )


def _apply_comparison(
    *,
    budget: int,
    comparison_path: Path,
    rows_by_id: Mapping[str, dict[str, Any]],
    aliases: Mapping[str, set[str]],
    model_index: Mapping[str, set[str]],
    explicit_arm_trials: Mapping[str, str],
    training_run_root: Path,
    task_matrices: dict[int, str],
) -> None:
    if budget not in VALIDATION_BUDGETS:
        raise ValueError(
            f"Validation budget must be one of {VALIDATION_BUDGETS}, got {budget}"
        )
    comparison = _read_json_object(comparison_path, label="comparison")
    task_count = comparison.get("task_count")
    if task_count != budget:
        raise ValueError(
            f"Comparison budget {budget} does not match task_count={task_count!r}: "
            f"{comparison_path}"
        )
    arms = comparison.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError(f"Comparison has no arms array: {comparison_path}")
    task_matrix_sha256 = str(comparison.get("task_matrix_sha256") or "")
    if not task_matrix_sha256:
        raise ValueError(f"Comparison has no task_matrix_sha256: {comparison_path}")
    existing_task_matrix = task_matrices.get(budget)
    if existing_task_matrix is not None and existing_task_matrix != task_matrix_sha256:
        raise ValueError(
            f"Budget {budget} comparisons use different task matrices: "
            f"{existing_task_matrix} != {task_matrix_sha256}"
        )
    task_matrices[budget] = task_matrix_sha256
    ranking = comparison.get("eligible_ranking")
    if not isinstance(ranking, list):
        raise ValueError(f"Comparison has no eligible_ranking array: {comparison_path}")
    rank_by_name = {
        str(item.get("name")): index
        for index, item in enumerate(ranking, 1)
        if isinstance(item, Mapping) and item.get("name")
    }

    unmatched: list[str] = []
    for arm in arms:
        if not isinstance(arm, Mapping):
            raise ValueError(f"Comparison arm must be an object: {comparison_path}")
        arm_name = str(arm.get("name") or "")
        arm_path = str(arm.get("path") or "")
        if not arm_name or not arm_path:
            raise ValueError(f"Comparison arm has no name/path: {comparison_path}")
        # A trained candidate can become the control for a later stage at the
        # same budget (for example, selected DMPO -> DEPO). Its candidate
        # measurement is the informative ledger entry; do not overwrite it
        # with a self-comparison whose paired deltas are all zero.
        if arm_name == comparison.get("baseline"):
            continue
        trial_id = _match_arm(
            arm_name=arm_name,
            arm_path=arm_path,
            trials_by_id=rows_by_id,
            aliases=aliases,
            model_index=model_index,
            explicit_arm_trials=explicit_arm_trials,
            training_run_root=training_run_root,
        )
        if trial_id is None:
            unmatched.append(arm_name)
            continue
        _add_measurement(
            rows_by_id[trial_id],
            budget=budget,
            comparison_path=comparison_path,
            comparison=comparison,
            arm=arm,
            rank=rank_by_name.get(arm_name, ""),
        )
    if unmatched:
        raise ValueError(
            f"Comparison arms do not match training trials: {', '.join(sorted(unmatched))}. "
            "Name each arm with its canonical trial ID/name, keep its evaluation "
            "artifacts beside the training run, or pass --arm-trial ARM=TRIAL_ID."
        )


def _set_latest_results(rows: Sequence[dict[str, Any]]) -> None:
    for row in rows:
        measured = [
            budget
            for budget in VALIDATION_BUDGETS
            if row[f"val_{budget}_comparison_path"]
        ]
        if not measured:
            continue
        latest = max(measured)
        row["latest_budget"] = latest
        if row[f"val_{latest}_selected"]:
            row["latest_result"] = "selected"
        elif row[f"val_{latest}_selection_eligible"]:
            row["latest_result"] = "eligible_not_selected"
        else:
            row["latest_result"] = "ineligible"


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with staging.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row.get(field, "")) for field in FIELDNAMES})
        os.replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def build_trial_record(
    *,
    run_root: str | Path,
    comparisons: Sequence[tuple[int, str | Path]] = (),
    output: str | Path | None = None,
    arm_trials: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build a deterministic trial ledger and return its rows."""

    root = Path(run_root).expanduser().resolve()
    rows, model_index = _discover_trials(root)
    rows_by_id = {str(row["trial_id"]): row for row in rows}
    aliases = _trial_alias_index(rows)
    explicit_arm_trials = dict(arm_trials or {})
    task_matrices: dict[int, str] = {}
    for budget, comparison in comparisons:
        _apply_comparison(
            budget=budget,
            comparison_path=Path(comparison).expanduser(),
            rows_by_id=rows_by_id,
            aliases=aliases,
            model_index=model_index,
            explicit_arm_trials=explicit_arm_trials,
            training_run_root=root,
            task_matrices=task_matrices,
        )
    _set_latest_results(rows)
    output_path = (
        Path(output).expanduser()
        if output is not None
        else root / "experiments" / "trial-record.csv"
    )
    _write_csv(output_path, rows)
    return rows


def _parse_comparison(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected BUDGET=COMPARISON_JSON")
    raw_budget, raw_path = value.split("=", 1)
    try:
        budget = int(raw_budget)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"Invalid validation budget: {raw_budget!r}") from error
    if budget not in VALIDATION_BUDGETS:
        raise argparse.ArgumentTypeError(
            f"Validation budget must be one of {VALIDATION_BUDGETS}"
        )
    if not raw_path.strip():
        raise argparse.ArgumentTypeError("Comparison path must not be empty")
    return budget, Path(raw_path).expanduser()


def _parse_arm_trial(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected ARM=TRIAL_ID")
    arm, trial_id = (item.strip() for item in value.split("=", 1))
    if not arm or not trial_id:
        raise argparse.ArgumentTypeError("ARM and TRIAL_ID must not be empty")
    return arm, trial_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument(
        "--comparison",
        action="append",
        default=[],
        type=_parse_comparison,
        metavar="BUDGET=JSON",
        help="Add a 100-, 200-, or 500-task comparison; repeat as needed.",
    )
    parser.add_argument(
        "--arm-trial",
        action="append",
        default=[],
        type=_parse_arm_trial,
        metavar="ARM=TRIAL_ID",
        help="Resolve an arm name that cannot be matched from evaluation metadata.",
    )
    parser.add_argument(
        "--output",
        help="Defaults to <run-root>/experiments/trial-record.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    arm_trials = dict(args.arm_trial)
    if len(arm_trials) != len(args.arm_trial):
        raise ValueError("Duplicate --arm-trial arm names are not allowed")
    rows = build_trial_record(
        run_root=args.run_root,
        comparisons=args.comparison,
        output=args.output,
        arm_trials=arm_trials,
    )
    output = (
        Path(args.output).expanduser()
        if args.output
        else Path(args.run_root).expanduser() / "experiments" / "trial-record.csv"
    )
    print(f"Wrote {len(rows)} trial rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
