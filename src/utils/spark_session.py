"""Spark session factory for CloudIQ with Delta Lake support.

Uses ``delta-spark``'s :func:`configure_spark_with_delta_pip` so the Delta JAR
is resolved at session creation (Correction 12), in addition to the Delta SQL
extension and catalog configuration. No Spark session is created at import time;
the JVM only starts when :func:`get_spark_session` is called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from src.utils.config import ConfigLoader
from src.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

_logger = get_logger("utils.spark_session")


def get_spark_session(
    config: ConfigLoader,
    app_name: Optional[str] = None,
) -> "SparkSession":
    """Create and return a local Delta-enabled :class:`SparkSession`.

    Imports of ``pyspark`` and ``delta`` are deferred to call time so that
    importing this module never starts a JVM or requires the Delta JAR.

    Args:
        config: Loaded :class:`ConfigLoader` providing Spark settings.
        app_name: Optional override for the Spark application name.

    Returns:
        A configured :class:`SparkSession` running in ``local[*]`` mode.
    """
    from delta import configure_spark_with_delta_pip
    from pyspark.sql import SparkSession

    app_name = app_name or config.get("spark.app_name", "CloudIQ")

    builder = (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config(
            "spark.sql.extensions",
            "io.delta.sql.DeltaSparkSessionExtension",
        )
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        .config("spark.driver.memory", config.get("spark.driver_memory", "4g"))
        .config(
            "spark.executor.memory",
            config.get("spark.executor_memory", "4g"),
        )
        .config(
            "spark.sql.shuffle.partitions",
            str(config.get("spark.shuffle_partitions", 8)),
        )
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    _logger.info(
        "SparkSession started \u2014 app={}, version={}",
        app_name,
        spark.version,
    )
    return spark
