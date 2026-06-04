"""Export Inspect ``.eval`` logs to the long-format table the TEE R pipeline needs.

The ``totalevalerror`` R package decomposes variance from a long-format data
frame passed to ``tee_design()``: one row per (item × prompt variant ×
replication) observation, with the design facets as columns. This module reads
one or more ``.eval`` logs produced by a TEE eval (e.g. ``mmlu_tee``) and emits
that table as a CSV; an optional step shells out to ``scripts/long_csv_to_rda.R``
to also write the ``.rda`` the package bundles its example data as.

The exporter is generic across TEE evals: it reads the standard sample-metadata
keys (``item_id``, ``variant_id``, ``category``, ``language``, ``subject``), the
model from the log, the temperature from the resolved generation config, and the
replication index from the Inspect epoch. Scores map to a binary outcome, with
parse failures (``NOANSWER``) becoming ``NA`` so ``tee_design()`` drops them —
matching the paper's "exclude unparseable" rule.

    uv run python -m utils.tee_export --logs logs/ --out exports/tee_long.csv --rda
"""

from __future__ import annotations

import argparse
import glob
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log
from inspect_ai.scorer import CORRECT, INCORRECT

logger = logging.getLogger(__name__)

# Long-format columns. The first seven are the tee_design() facets (matching the
# bundled mmlu_pilot schema); language and subject are extra bookkeeping columns
# that tee_design() ignores but which let the R side slice cells and report.
TEE_COLUMNS = [
    "item_id",
    "category",
    "variant_id",
    "sut_model",
    "temperature",
    "replication",
    "outcome",
    "language",
    "subject",
]

DEFAULT_LOGS = "logs"
DEFAULT_OUT = "exports/tee_long.csv"
RDA_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "long_csv_to_rda.R"


def score_to_outcome(value: Any) -> float | None:
    """Map an Inspect score value to a binary outcome.

    ``CORRECT`` → 1.0, ``INCORRECT`` → 0.0, anything else (``NOANSWER`` / parse
    failure) → ``None``, which becomes ``NA`` in the CSV and is dropped by
    ``tee_design()``.
    """
    if value == CORRECT:
        return 1.0
    if value == INCORRECT:
        return 0.0
    return None


def resolve_temperature(log: EvalLog) -> float | None:
    """Return the generation temperature used for a run.

    Prefers the resolved plan config (reflects CLI overrides); falls back to the
    model's generate config.
    """
    plan = getattr(log, "plan", None)
    plan_config = getattr(plan, "config", None)
    if plan_config is not None and plan_config.temperature is not None:
        return float(plan_config.temperature)
    model_config = getattr(log.eval, "model_generate_config", None)
    if model_config is not None and model_config.temperature is not None:
        return float(model_config.temperature)
    return None


def log_to_rows(log: EvalLog) -> list[dict[str, Any]]:
    """Convert one ``EvalLog`` to long-format rows (one per sample observation).

    Samples missing the TEE metadata keys, or with no score, are skipped with a
    warning. Uses the first scorer's value (TEE evals score with a single scorer).
    """
    model = log.eval.model
    temperature = resolve_temperature(log)
    rows: list[dict[str, Any]] = []
    for sample in log.samples or []:
        scores = sample.scores or {}
        metadata = sample.metadata or {}
        if "item_id" not in metadata or "variant_id" not in metadata:
            logger.warning("Sample %s lacks TEE metadata; skipping.", sample.id)
            continue
        if not scores:
            logger.warning("Sample %s has no score; skipping.", sample.id)
            continue
        score = next(iter(scores.values()))
        rows.append(
            {
                "item_id": metadata.get("item_id"),
                "category": metadata.get("category"),
                "variant_id": metadata.get("variant_id"),
                "sut_model": model,
                "temperature": temperature,
                "replication": sample.epoch - 1,
                "outcome": score_to_outcome(score.value),
                "language": metadata.get("language"),
                "subject": metadata.get("subject"),
            }
        )
    return rows


def _resolve_log_paths(entries: list[str]) -> list[str]:
    """Expand log entries (directories, globs, or files) to concrete log paths."""
    paths: list[str] = []
    for entry in entries:
        if Path(entry).is_dir():
            paths.extend(info.name for info in list_eval_logs(entry))
        elif any(char in entry for char in "*?["):
            paths.extend(sorted(glob.glob(entry)))
        elif Path(entry).is_file():
            paths.append(entry)
        else:
            logger.warning("Log entry not found, skipping: %s", entry)
    return paths


def export_logs(log_entries: list[str], out_csv: str | Path) -> pd.DataFrame:
    """Read logs, build the combined long-format frame, and write it to CSV.

    Raises:
        FileNotFoundError: if no logs are found at ``log_entries``.
    """
    paths = _resolve_log_paths(log_entries)
    if not paths:
        raise FileNotFoundError(f"No .eval logs found at: {log_entries}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(log_to_rows(read_eval_log(path)))

    frame = pd.DataFrame(rows, columns=TEE_COLUMNS)
    frame["outcome"] = pd.to_numeric(frame["outcome"], errors="coerce")
    frame["replication"] = pd.to_numeric(frame["replication"], errors="coerce").astype(
        "Int64"
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    return frame


def parse_failure_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-cell counts of observations, parse failures, and parse rate.

    A cell is a (``sut_model``, ``language``) pair; a parse failure is a row with
    a missing (``NA``) outcome.
    """
    annotated = frame.assign(parse_failed=frame["outcome"].isna())
    report = (
        annotated.groupby(["sut_model", "language"], dropna=False)
        .agg(n=("outcome", "size"), parse_failures=("parse_failed", "sum"))
        .reset_index()
    )
    report["parse_rate"] = 1.0 - report["parse_failures"] / report["n"]
    return report


def write_rda(csv_path: str | Path, rda_path: str | Path, name: str) -> None:
    """Convert a long-format CSV to ``.rda`` via the base-R converter script.

    Raises:
        RuntimeError: if ``Rscript`` is not on PATH.
    """
    if shutil.which("Rscript") is None:
        raise RuntimeError("Rscript not found on PATH; install R or omit --rda.")
    subprocess.run(
        ["Rscript", str(RDA_SCRIPT), str(csv_path), str(rda_path), name],
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: export ``.eval`` logs to long-format CSV (and optional .rda)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--logs",
        nargs="+",
        default=[DEFAULT_LOGS],
        help="Log directories, globs, or files to read (default: logs/).",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Output CSV path (default: exports/tee_long.csv).",
    )
    parser.add_argument(
        "--rda",
        action="store_true",
        help="Also write a .rda alongside the CSV (requires Rscript).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="R object name inside the .rda (default: the --out file stem).",
    )
    args = parser.parse_args(argv)

    frame = export_logs(args.logs, args.out)
    logger.info("Wrote %d rows to %s", len(frame), args.out)
    logger.info("\nParse-failure report (per cell):\n%s", parse_failure_report(frame))

    if args.rda:
        rda_path = Path(args.out).with_suffix(".rda")
        name = args.name or Path(args.out).stem
        write_rda(args.out, rda_path, name)
        logger.info("Wrote %s (R object '%s')", rda_path, name)


if __name__ == "__main__":
    main()
