"""Spark session factory for CloudIQ with Delta Lake support.

Uses ``delta-spark``'s :func:`configure_spark_with_delta_pip` so the Delta JAR
is resolved at session creation (Correction 12), in addition to the Delta SQL
extension and catalog configuration. No Spark session is created at import time;
the JVM only starts when :func:`get_spark_session` is called.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from src.utils.config import ConfigLoader
from src.utils.logger import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import SparkSession

_logger = get_logger("utils.spark_session")
_WINDOWS_HADOOP_CANDIDATES = (Path("C:/hadoop"), Path("C:/tools/hadoop"))


def _configure_windows_hadoop_home() -> Optional[Path]:
    """Set Hadoop home on Windows when a local winutils.exe is available."""
    if os.name != "nt":
        return None

    configured = [
        Path(value)
        for value in (
            os.environ.get("HADOOP_HOME"),
            os.environ.get("hadoop.home.dir"),
        )
        if value
    ]

    for hadoop_home in [*configured, *_WINDOWS_HADOOP_CANDIDATES]:
        hadoop_bin = hadoop_home / "bin"
        if (hadoop_bin / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(hadoop_home)
            os.environ["hadoop.home.dir"] = str(hadoop_home)
            current_path = os.environ.get("PATH", "")
            path_parts = [p for p in current_path.split(os.pathsep) if p]
            if str(hadoop_bin) not in path_parts:
                os.environ["PATH"] = os.pathsep.join(
                    [str(hadoop_bin), *path_parts]
                )
            _logger.debug("Using Hadoop home for local Spark: {}", hadoop_home)
            return hadoop_home

    _logger.warning(
        "HADOOP_HOME is unset and winutils.exe was not found; "
        "local Spark startup may fail on Windows."
    )
    return None


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
    hadoop_home = _configure_windows_hadoop_home()

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

    if hadoop_home is not None:
        hadoop_home_arg = hadoop_home.as_posix()
        hadoop_bin_arg = (hadoop_home / "bin").as_posix()
        java_options = (
            f"-Dhadoop.home.dir={hadoop_home_arg} "
            f"-Djava.library.path={hadoop_bin_arg}"
        )
        builder = builder.config(
            "spark.driver.extraJavaOptions",
            java_options,
        ).config(
            "spark.executor.extraJavaOptions",
            java_options,
        )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    _logger.info(
        "SparkSession started \u2014 app={}, version={}",
        app_name,
        spark.version,
    )
    return spark
