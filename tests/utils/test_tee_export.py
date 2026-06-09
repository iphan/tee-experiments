"""Tests for the .eval log -> TEE long-format / .rda exporter."""

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import CORRECT, INCORRECT, NOANSWER
from inspect_ai.solver import generate

from mmlu.mmlu import letter_choice, mmlu_tee
from utils.tee_export import (
    TEE_COLUMNS,
    export_logs,
    log_to_rows,
    parse_failure_details,
    parse_failure_report,
    parse_failure_summary,
    resolve_temperature,
    score_to_outcome,
    write_rda,
)


def _synthetic_log(target: str, content: str, *, epochs: int, log_dir: Path) -> EvalLog:
    """Run a one-sample task via mockllm that always emits ``content``."""
    sample = Sample(
        input="q",
        target=target,
        id="math_0001__V_1",
        metadata={
            "item_id": "math_0001",
            "variant_id": "V_1",
            "category": "STEM",
            "language": "en",
            "subject": "high_school_mathematics",
        },
    )
    task = Task(
        dataset=MemoryDataset([sample]),
        solver=generate(),
        scorer=letter_choice(),
    )
    outputs = [
        ModelOutput.from_content(model="mockllm/model", content=content)
    ] * epochs
    [log] = eval(
        task,
        model=get_model("mockllm/model", custom_outputs=outputs),
        epochs=epochs,
        log_dir=str(log_dir),
    )
    return log


class TestScoreToOutcome:
    def test_correct_maps_to_one(self) -> None:
        # given a CORRECT value / when mapped / then 1.0
        assert score_to_outcome(CORRECT) == 1.0

    def test_incorrect_maps_to_zero(self) -> None:
        # given an INCORRECT value / when mapped / then 0.0
        assert score_to_outcome(INCORRECT) == 0.0

    @pytest.mark.parametrize("value", [NOANSWER, "unexpected", None])
    def test_other_maps_to_none(self, value: object) -> None:
        # given a non-binary value / when mapped / then None (-> NA, dropped)
        assert score_to_outcome(value) is None


class TestLogToRows:
    def test_correct_answer_yields_outcome_one(self, tmp_path: Path) -> None:
        # given a model that answers the target / when exported
        log = _synthetic_log("B", "B", epochs=1, log_dir=tmp_path)
        rows = log_to_rows(log)
        # then the single row scores 1.0
        assert len(rows) == 1
        assert rows[0]["outcome"] == 1.0

    def test_wrong_answer_yields_outcome_zero(self, tmp_path: Path) -> None:
        # given a parseable but wrong answer / when exported / then 0.0
        rows = log_to_rows(_synthetic_log("A", "B", epochs=1, log_dir=tmp_path))
        assert rows[0]["outcome"] == 0.0

    def test_unparseable_yields_none_outcome(self, tmp_path: Path) -> None:
        # given an unparseable answer / when exported / then None (NA)
        rows = log_to_rows(_synthetic_log("B", "no idea", epochs=1, log_dir=tmp_path))
        assert rows[0]["outcome"] is None

    def test_row_carries_design_facets(self, tmp_path: Path) -> None:
        # given a run / when a row is built / then it carries the design facets
        rows = log_to_rows(_synthetic_log("B", "B", epochs=1, log_dir=tmp_path))
        row = rows[0]
        assert row["item_id"] == "math_0001"
        assert row["variant_id"] == "V_1"
        assert row["category"] == "STEM"
        assert row["language"] == "en"
        assert row["subject"] == "high_school_mathematics"
        assert row["sut_model"] == "mockllm/model"
        assert row["replication"] == 0  # epoch 1 -> replication 0

    def test_epochs_map_to_zero_indexed_replications(self, tmp_path: Path) -> None:
        # given three epochs / when exported / then replications are 0,1,2
        rows = log_to_rows(_synthetic_log("B", "B", epochs=3, log_dir=tmp_path))
        assert sorted(r["replication"] for r in rows) == [0, 1, 2]


class TestMmluIntegration:
    def test_real_task_columns_temperature_and_parse_report(
        self, tmp_path: Path
    ) -> None:
        # given the real mmlu_tee task (bakes temperature=0.7) run on mockllm
        task = mmlu_tee(n_items=2, n_variants=2)  # epochs default 3 -> 2*2*3 = 12
        outputs = [
            ModelOutput.from_content(model="mockllm/model", content="no idea")
        ] * 200
        [log] = eval(
            task,
            model=get_model("mockllm/model", custom_outputs=outputs),
            log_dir=str(tmp_path),
        )
        # when exported
        rows = log_to_rows(log)
        # then temperature, replication range, and parse failures are correct
        assert resolve_temperature(log) == 0.7
        assert len(rows) == 12
        assert {r["replication"] for r in rows} == {0, 1, 2}
        assert all(r["temperature"] == 0.7 for r in rows)
        assert all(r["outcome"] is None for r in rows)  # unparseable -> NA


class TestExportLogs:
    def test_writes_csv_with_expected_columns(self, tmp_path: Path) -> None:
        # given a directory holding two logs
        logs_dir = tmp_path / "logs"
        _synthetic_log("B", "B", epochs=1, log_dir=logs_dir)
        _synthetic_log("A", "B", epochs=1, log_dir=logs_dir)
        # when exported to CSV
        out_csv = tmp_path / "exports" / "tee_long.csv"
        frame = export_logs([str(logs_dir)], out_csv)
        # then both logs' rows are combined, and the CSV holds only TEE_COLUMNS
        assert out_csv.exists()
        assert list(pd.read_csv(out_csv).columns) == TEE_COLUMNS
        assert len(frame) == 2

    def test_csv_excludes_report_columns(self, tmp_path: Path) -> None:
        # given a run that produces a completion
        logs_dir = tmp_path / "logs"
        _synthetic_log("B", "B", epochs=1, log_dir=logs_dir)
        out_csv = tmp_path / "out.csv"
        # when exported
        frame = export_logs([str(logs_dir)], out_csv)
        # then the frame carries reporting columns but the CSV does not
        assert {"explanation", "stop_reason"}.issubset(frame.columns)
        assert "explanation" not in pd.read_csv(out_csv).columns
        assert "stop_reason" not in pd.read_csv(out_csv).columns

    def test_parse_failure_report_counts_na_outcomes(self, tmp_path: Path) -> None:
        # given one correct and one unparseable observation
        logs_dir = tmp_path / "logs"
        _synthetic_log("B", "B", epochs=1, log_dir=logs_dir)
        _synthetic_log("B", "no idea", epochs=1, log_dir=logs_dir)
        frame = export_logs([str(logs_dir)], tmp_path / "out.csv")
        # when the per-cell report is built / then it counts the one failure
        report = parse_failure_report(frame)
        assert int(report["n"].sum()) == 2
        assert int(report["parse_failures"].sum()) == 1


class TestParseFailureDiagnostics:
    def test_details_list_failing_completion(self, tmp_path: Path) -> None:
        # given one parseable and one unparseable observation
        logs_dir = tmp_path / "logs"
        _synthetic_log("B", "B", epochs=1, log_dir=logs_dir)
        _synthetic_log("B", "no idea here", epochs=1, log_dir=logs_dir)
        frame = export_logs([str(logs_dir)], tmp_path / "out.csv")
        # when the detail listing is built
        details = parse_failure_details(frame)
        # then only the failure is listed, with its completion as the error message
        assert len(details) == 1
        assert details.iloc[0]["item_id"] == "math_0001"
        assert details.iloc[0]["completion"] == "no idea here"

    def test_details_collapse_and_truncate_completion(self, tmp_path: Path) -> None:
        # given an unparseable multi-line completion longer than the preview budget
        logs_dir = tmp_path / "logs"
        long_text = "To solve this\nwe must\n" + "x " * 200
        _synthetic_log("B", long_text, epochs=1, log_dir=logs_dir)
        frame = export_logs([str(logs_dir)], tmp_path / "out.csv")
        # when the detail listing is built
        completion = parse_failure_details(frame).iloc[0]["completion"]
        # then it is a single truncated line
        assert "\n" not in completion
        assert completion.startswith("To solve this we must")
        assert completion.endswith("…")
        assert len(completion) == 100

    def test_summary_counts_failures_by_stop_reason(self, tmp_path: Path) -> None:
        # given two unparseable observations from a mockllm run (natural stop)
        logs_dir = tmp_path / "logs"
        _synthetic_log("B", "no idea", epochs=2, log_dir=logs_dir)
        frame = export_logs([str(logs_dir)], tmp_path / "out.csv")
        # when summarised by stop_reason / then both failures are counted
        summary = parse_failure_summary(frame)
        assert int(summary.sum()) == 2


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")
class TestRdaConversion:
    def test_rda_round_trips_row_count(self, tmp_path: Path) -> None:
        # given an exported CSV
        logs_dir = tmp_path / "logs"
        _synthetic_log("B", "B", epochs=2, log_dir=logs_dir)
        csv_path = tmp_path / "tee_long.csv"
        frame = export_logs([str(logs_dir)], csv_path)
        # when converted to .rda and loaded back in R
        rda_path = tmp_path / "tee_long.rda"
        write_rda(csv_path, rda_path, "tee_long")
        assert rda_path.exists()
        result = subprocess.run(
            ["Rscript", "-e", f'load("{rda_path}"); cat(nrow(tee_long))'],
            capture_output=True,
            text=True,
            check=True,
        )
        # then the row count matches the exported frame
        assert int(result.stdout.strip()) == len(frame)


def test_no_logs_found_raises(tmp_path: Path) -> None:
    # given an empty directory / when exported / then a clear error
    with pytest.raises(FileNotFoundError, match="No .eval logs found"):
        export_logs([str(tmp_path / "empty")], tmp_path / "out.csv")
