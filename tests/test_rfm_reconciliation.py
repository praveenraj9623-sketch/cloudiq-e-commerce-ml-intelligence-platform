"""RFM reconciliation checks for local dashboard and Databricks notebook logic."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

CANONICAL_RFM_TOTALS = {
    "Champion": 15321,
    "Loyal": 27618,
    "Potential": 29922,
    "At Risk": 19133,
    "Lost": 4102,
}
CUSTOMER_PROFILE_TOTAL = 96096


def _active_delta_adds(delta_path: Path) -> list[dict]:
    """Return active Delta add actions by replaying JSON transaction logs."""
    active: dict[str, dict] = {}
    for log_file in sorted((delta_path / "_delta_log").glob("*.json")):
        with log_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                action = json.loads(line)
                if "add" in action:
                    active[action["add"]["path"]] = action["add"]
                elif "remove" in action:
                    active.pop(action["remove"]["path"], None)
    return [active[path] for path in sorted(active)]


def _read_delta_columns(delta_path: Path, columns: list[str]) -> pd.DataFrame:
    """Read selected active local Delta parquet columns into pandas."""
    frames: list[pd.DataFrame] = []
    for add in _active_delta_adds(delta_path):
        frame = (
            pq.ParquetFile(delta_path / Path(unquote(add["path"])))
            .read(columns=columns)
            .to_pandas()
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_local_dashboard_rfm_distribution_matches_canonical_totals() -> None:
    """Dashboard RFM distribution matches the canonical local Gold output."""
    distribution = pd.read_csv(ROOT / "data" / "dashboard" / "rfm_segment_distribution.csv")
    actual = dict(zip(distribution["segment_label"], distribution["customers"]))

    assert actual == CANONICAL_RFM_TOTALS


def test_rfm_segment_labels_sum_to_customer_profile_total() -> None:
    """Canonical segment totals sum to the Silver customer profile total."""
    distribution = pd.read_csv(ROOT / "data" / "dashboard" / "rfm_segment_distribution.csv")
    quality = json.loads(
        (ROOT / "data" / "dashboard" / "data_quality_summary.json").read_text(
            encoding="utf-8"
        )
    )
    customer_profile_total = quality["pipeline_validation"]["silver_row_counts"][
        "customer_profile"
    ]

    assert int(distribution["customers"].sum()) == CUSTOMER_PROFILE_TOTAL
    assert int(distribution["customers"].sum()) == customer_profile_total


def test_gold_rfm_assigns_exactly_one_segment_per_customer() -> None:
    """Every customer in canonical Gold RFM receives exactly one segment."""
    rfm = _read_delta_columns(
        ROOT / "data" / "gold" / "rfm_segments",
        ["customer_unique_id", "segment_label"],
    )

    assert len(rfm) == CUSTOMER_PROFILE_TOTAL
    assert rfm["customer_unique_id"].nunique() == CUSTOMER_PROFILE_TOTAL
    assert rfm["segment_label"].isna().sum() == 0


def test_databricks_rfm_notebook_uses_canonical_threshold_logic() -> None:
    """Databricks Free Edition notebook mirrors local canonical RFM rules."""
    notebook = (
        ROOT
        / "databricks"
        / "free_edition"
        / "CloudIQ_Databricks_Free_Edition"
        / "cloudiq_03_gold_final_managed_delta.py"
    ).read_text(encoding="utf-8")

    assert "F.ntile(5).over(frequency_window)" in notebook
    assert 'F.col("rfm_total") >= 13' in notebook
    assert 'F.col("rfm_total") >= 10' in notebook
    assert 'F.col("rfm_total") >= 7' in notebook
    assert 'RFM_LOGIC_VERSION = "local_gold_canonical_v1"' in notebook
    assert '"Champion": 15321' in notebook
    assert '"Loyal": 27618' in notebook
    assert '"Potential": 29922' in notebook
    assert '"At Risk": 19133' in notebook
    assert '"Lost": 4102' in notebook
    assert "EXPECTED_RFM_TOTAL_ROWS = 96096" in notebook
    assert 'withColumn("rfm_logic_version", F.lit(RFM_LOGIC_VERSION))' in notebook
    assert "Databricks RFM parity validation failed" in notebook
    assert "rfm_unique_customers" in notebook
    assert "rfm_null_segments" in notebook
    assert "total_orders\") >= 3" not in notebook
    assert "_recency_percent_rank" not in notebook
    assert "_monetary_percent_rank" not in notebook
    assert "percent_rank()" not in notebook
    assert "repeat_customer_flag" not in notebook
