"""Verify the local CloudIQ foundation with a tiny Delta smoke test."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from pyspark.sql import SparkSession

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import ConfigLoader  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.spark_session import get_spark_session  # noqa: E402

_LOGGER = get_logger("scripts.verify_foundation")


def run_smoke_test() -> bool:
    """Run a local Delta write/read verification and return success."""
    os.chdir(_ROOT)
    spark: SparkSession | None = None
    smoke_path = _ROOT / "data" / "_smoke_delta"

    try:
        config = ConfigLoader(
            str(_ROOT / "config.yaml"),
            env_path=str(_ROOT / ".env"),
        )
        config.validate(strict=False)

        spark = get_spark_session(
            config,
            app_name="CloudIQ-Foundation-Smoke",
        )
        rows = [(1, "alpha"), (2, "beta")]
        spark.createDataFrame(rows, ["id", "label"]).write.format(
            "delta"
        ).mode("overwrite").save(str(smoke_path))

        row_count = spark.read.format("delta").load(str(smoke_path)).count()
        if row_count != 2:
            _LOGGER.error(
                "Delta smoke row count mismatch: expected=2 actual={}",
                row_count,
            )
            print(
                "FAIL: Delta smoke test expected 2 rows "
                f"but read {row_count}"
            )
            return False

        print(
            "PASS: Delta smoke test wrote and read 2 rows at "
            f"{smoke_path}"
        )
        return True
    except Exception as exc:  # noqa: BLE001 - command reports any failure
        _LOGGER.opt(exception=True).error("Foundation verification failed")
        print(f"FAIL: Delta smoke test failed: {exc}")
        return False
    finally:
        if spark is not None:
            spark.stop()
            _LOGGER.info("SparkSession stopped")


def main() -> int:
    """Return a process exit code for the foundation verification."""
    return 0 if run_smoke_test() else 1


if __name__ == "__main__":
    raise SystemExit(main())
