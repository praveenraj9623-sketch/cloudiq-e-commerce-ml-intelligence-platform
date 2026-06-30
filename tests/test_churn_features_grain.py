"""Data-contract checks for the local Gold churn feature table grain."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
CHURN_PATH = ROOT / "data" / "gold" / "churn_features"
NATURAL_KEY = ["snapshot_date", "customer_unique_id"]

EXPECTED_ROW_COUNT = 453617
EXPECTED_CUSTOMER_COUNT = 69125
EXPECTED_SNAPSHOT_COUNT = 17
EXPECTED_COLUMNS = {
    "snapshot_date",
    "customer_unique_id",
    "is_churned",
    "total_orders",
    "total_revenue",
    "avg_order_value",
    "first_purchase_timestamp",
    "last_purchase_timestamp",
    "recency_days",
    "customer_age_days",
    "avg_review_score",
    "last_review_creation_date",
    "late_delivery_rate",
    "last_delivery_date",
    "purchase_frequency_30d",
    "log_recency",
    "log_total_revenue",
    "revenue_per_order",
}


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


def _read_delta_table(delta_path: Path) -> pd.DataFrame:
    """Read active local Delta parquet files into pandas for lightweight tests."""
    frames: list[pd.DataFrame] = []
    for add in _active_delta_adds(delta_path):
        partition_values = add.get("partitionValues", {})
        frame = pq.ParquetFile(delta_path / Path(unquote(add["path"]))).read().to_pandas()
        for key, value in partition_values.items():
            frame[key] = value
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_churn_features_uses_customer_snapshot_natural_key() -> None:
    """Every churn row is unique by the documented customer snapshot key."""
    churn = _read_delta_table(CHURN_PATH)

    assert len(churn) == EXPECTED_ROW_COUNT
    assert churn["customer_unique_id"].nunique() == EXPECTED_CUSTOMER_COUNT
    assert churn["snapshot_date"].nunique() == EXPECTED_SNAPSHOT_COUNT
    assert churn[NATURAL_KEY].isna().sum().sum() == 0
    assert churn[NATURAL_KEY].drop_duplicates().shape[0] == len(churn)


def test_churn_features_expected_columns_remain_present() -> None:
    """The snapshot churn table keeps the expected feature and label columns."""
    churn = _read_delta_table(CHURN_PATH)

    assert EXPECTED_COLUMNS.issubset(churn.columns)
    assert churn["is_churned"].isna().sum() == 0
