"""Focused Bronze reviews ingestion test for quoted multiline comments."""

from __future__ import annotations

from pathlib import Path

from src.processing.bronze import BronzeLayer
from src.utils.config import ConfigLoader
from src.utils.spark_session import get_spark_session


def test_reviews_ingestion_preserves_multiline_comment_as_one_row(
    tmp_path: Path,
) -> None:
    """A quoted multiline review comment remains one logical Spark row."""
    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    raw_dir.mkdir()
    csv_path = raw_dir / "reviews.csv"
    csv_path.write_text(
        "\n".join(
            [
                "review_id,order_id,review_score,review_comment_title,"
                "review_comment_message,review_creation_date,"
                "review_answer_timestamp",
                'r1,o1,5,Great,"first line',
                'second line",2018-01-01 00:00:00,2018-01-02 00:00:00',
            ]
        ),
        encoding="utf-8",
    )

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "spark:",
                '  app_name: "CloudIQ-Test-Reviews-Multiline"',
                '  driver_memory: "1g"',
                '  executor_memory: "1g"',
                "  shuffle_partitions: 1",
                "paths:",
                f'  raw: "{raw_dir.as_posix()}"',
                f'  bronze: "{bronze_dir.as_posix()}"',
                "olist_files:",
                '  reviews: "reviews.csv"',
            ]
        ),
        encoding="utf-8",
    )
    config = ConfigLoader(str(config_path), env_path="nonexistent.env")
    spark = get_spark_session(config, app_name="CloudIQ-Test-Reviews-Multiline")
    try:
        result = BronzeLayer(spark, config).ingest_reviews()
        assert result["status"] == "SUCCESS"
        assert result["rows"] == 1
        bronze_count = (
            spark.read.format("delta").load(str(bronze_dir / "reviews")).count()
        )
        assert bronze_count == 1
    finally:
        spark.stop()
