#!/usr/bin/env python3
"""Verify that selected SWE-smith task refs exist in cached repository SIFs."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from debug_depo.apptainer_cache import swesmith_cache_images
from debug_depo.constants import (
    DEFAULT_SWESMITH_DATASET,
    DEFAULT_SWESMITH_DATASET_REVISION,
    DEFAULT_SWESMITH_SPLIT,
)
from debug_depo.data import load_swebench_tasks, read_instance_ids_file, select_tasks


DEFAULT_INSTANCE_IDS_FILE = (
    "data/splits/swesmith_validation_confirmatory_balanced_500_instance_ids.txt"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check that every selected SWE-smith instance ref resolves inside its "
            "cached repository image."
        )
    )
    parser.add_argument("--instance-ids-file", default=DEFAULT_INSTANCE_IDS_FILE)
    parser.add_argument("--dataset", default=DEFAULT_SWESMITH_DATASET)
    parser.add_argument("--dataset-revision", default=DEFAULT_SWESMITH_DATASET_REVISION)
    parser.add_argument("--split", default=DEFAULT_SWESMITH_SPLIT)
    parser.add_argument(
        "--sif-dir",
        default=os.getenv(
            "SWESMITH_APPTAINER_SIF_DIR",
            "data/apptainer/swesmith-sifs",
        ),
    )
    parser.add_argument(
        "--missing-output",
        help="Optional file in which to store unavailable instance IDs.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    requested_ids = read_instance_ids_file(args.instance_ids_file)
    tasks = select_tasks(
        load_swebench_tasks(
            args.dataset,
            args.split,
            revision=args.dataset_revision,
        ),
        instance_ids=requested_ids,
    )
    images = swesmith_cache_images(tasks, sif_dir=args.sif_dir)

    missing_ids: list[str] = []
    image_errors: list[str] = []
    check_script = (
        'for id in "$@"; do '
        'git -C /testbed rev-parse --verify --quiet "${id}^{commit}" >/dev/null '
        '|| git -C /testbed rev-parse --verify --quiet '
        '"refs/remotes/origin/${id}^{commit}" >/dev/null '
        '|| printf "%s\\n" "$id"; '
        "done"
    )

    for image in images:
        if not image.sif_path.is_file():
            image_errors.append(f"missing image: {image.sif_path}")
            continue
        completed = subprocess.run(
            [
                "apptainer",
                "exec",
                str(image.sif_path),
                "bash",
                "-lc",
                check_script,
                "preflight",
                *image.instance_ids,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or f"exit status {completed.returncode}"
            image_errors.append(f"{image.image_uri}: {detail}")
            continue
        missing_ids.extend(line for line in completed.stdout.splitlines() if line)

    if args.missing_output:
        output_path = Path(args.missing_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "".join(f"{instance_id}\n" for instance_id in missing_ids),
            encoding="utf-8",
        )

    print(f"Checked {len(requested_ids)} tasks across {len(images)} repository images.")
    print(f"Unavailable task refs: {len(missing_ids)}")
    for instance_id in missing_ids:
        print(instance_id)

    if image_errors:
        print(f"Image errors: {len(image_errors)}")
        for error in image_errors:
            print(error)
        return 2
    return 1 if missing_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
