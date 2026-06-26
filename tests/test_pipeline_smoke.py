"""Smoke tests that require no real Spark JVM or Delta JAR.

These tests import the pipeline modules and assert their public surface
(classes, methods, CLI parsing) without constructing a SparkSession. A full
Spark run is only exercised when ``data/raw`` is populated, which is out of
scope for unit tests.
"""

from __future__ import annotations

import importlib

import pytest


def test_processing_modules_import() -> None:
    """Bronze, silver, and gold modules import without starting Spark."""
    for module_name in (
        "src.processing.bronze",
        "src.processing.silver",
        "src.processing.gold",
    ):
        module = importlib.import_module(module_name)
        assert module is not None


def test_layer_classes_have_run_pipeline() -> None:
    """Each layer class exposes run_pipeline."""
    from src.processing.bronze import BronzeLayer
    from src.processing.gold import GoldLayer
    from src.processing.silver import SilverLayer

    for cls in (BronzeLayer, SilverLayer, GoldLayer):
        assert hasattr(cls, "run_pipeline")


def test_silver_seller_methods_present() -> None:
    """Silver exposes the corrected seller performance builder."""
    from src.processing.silver import SilverLayer

    assert hasattr(SilverLayer, "build_seller_performance")
    assert hasattr(SilverLayer, "build_master_orders")


def test_gold_demand_methods_present() -> None:
    """Gold exposes demand history and feature builders plus churn."""
    from src.processing.gold import GoldLayer

    for name in (
        "build_churn_features",
        "build_demand_history",
        "build_demand_forecast_features",
        "build_rfm_segments",
        "build_bi_revenue",
    ):
        assert hasattr(GoldLayer, name)


def test_run_pipeline_arg_parsing() -> None:
    """The runner parses --layer choices without creating Spark."""
    import run_pipeline

    for layer in ("bronze", "silver", "gold", "all"):
        args = run_pipeline._parse_args(["--layer", layer])
        assert args.layer == layer


def test_run_pipeline_rejects_unknown_layer() -> None:
    """An unknown --layer value is rejected by argparse."""
    import run_pipeline

    with pytest.raises(SystemExit):
        run_pipeline._parse_args(["--layer", "platinum"])
