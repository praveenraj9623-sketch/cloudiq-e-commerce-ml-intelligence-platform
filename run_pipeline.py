"""CloudIQ medallion pipeline runner.

Usage::

    python run_pipeline.py --layer bronze
    python run_pipeline.py --layer silver
    python run_pipeline.py --layer gold
    python run_pipeline.py --layer all

The Spark session is created only inside :func:`main`, never at import time.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console
from rich.table import Table

from src.utils.config import ConfigLoader
from src.utils.logger import get_logger

_logger = get_logger("run_pipeline")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="CloudIQ pipeline runner")
    parser.add_argument(
        "--layer",
        choices=["bronze", "silver", "gold", "all"],
        default="all",
        help="Which medallion layer(s) to run.",
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _run_layer(layer: str, spark, config: ConfigLoader) -> dict:
    """Run a single layer and return its results dict keyed by layer name."""
    from src.processing.bronze import BronzeLayer
    from src.processing.gold import GoldLayer
    from src.processing.silver import SilverLayer

    runners = {
        "bronze": lambda: BronzeLayer(spark, config).run_pipeline(),
        "silver": lambda: SilverLayer(spark, config).run_pipeline(),
        "gold": lambda: GoldLayer(spark, config).run_pipeline(),
    }
    return {layer: runners[layer]()}


def _collect_rows(all_results: dict) -> tuple[list[tuple], bool]:
    """Flatten nested results into table rows; return (rows, any_failed)."""
    rows: list[tuple] = []
    any_failed = False
    for layer, results in all_results.items():
        for table, info in results.items():
            if not isinstance(info, dict):
                continue
            status = info.get("status", "")
            if status == "FAILED":
                any_failed = True
            rows.append(
                (
                    layer,
                    info.get("table", table),
                    str(info.get("rows", "-")),
                    status or "-",
                    str(info.get("duration_s", "-")),
                )
            )
    return rows, any_failed


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = _parse_args(argv)
    config = ConfigLoader(args.config)
    config.validate(strict=False)

    from src.utils.spark_session import get_spark_session

    spark = get_spark_session(config)
    console = Console()
    all_results: dict = {}
    try:
        layers = (
            ["bronze", "silver", "gold"]
            if args.layer == "all"
            else [args.layer]
        )
        for layer in layers:
            all_results.update(_run_layer(layer, spark, config))

        rows, any_failed = _collect_rows(all_results)
        table = Table(title="CloudIQ Pipeline Results")
        for col in ("Layer", "Table", "Rows", "Status", "Duration(s)"):
            table.add_column(col)
        for row in rows:
            table.add_row(*row)
        console.print(table)

        if any_failed:
            console.print("[red]Pipeline completed with errors[/red]")
            return 1
        console.print("[green]Pipeline completed successfully[/green]")
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
