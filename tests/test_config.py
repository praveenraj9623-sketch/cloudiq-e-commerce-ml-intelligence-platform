"""Tests for :class:`ConfigLoader` placeholder resolution and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.utils.config import ConfigLoader


def _write_config(tmp_path: Path, body: str) -> str:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


def test_get_dot_notation(tmp_path: Path) -> None:
    """get() resolves nested keys via dot notation and returns defaults."""
    cfg = _write_config(
        tmp_path,
        """
        models:
          churn:
            churn_days_threshold: 90
        """,
    )
    loader = ConfigLoader(cfg, env_path="nonexistent.env")
    assert loader.get("models.churn.churn_days_threshold") == 90
    assert loader.get("models.missing.key", "fallback") == "fallback"


def test_default_interpolation(tmp_path: Path, monkeypatch) -> None:
    """${VAR:-default} resolves to the default when the env var is unset."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    cfg = _write_config(
        tmp_path,
        """
        mlflow:
          tracking_uri: "${MLFLOW_TRACKING_URI:-mlruns}"
        """,
    )
    loader = ConfigLoader(cfg, env_path="nonexistent.env")
    assert loader.get("mlflow.tracking_uri") == "mlruns"


def test_embedded_default_interpolation(tmp_path: Path, monkeypatch) -> None:
    """Placeholders embedded in larger strings resolve with defaults."""
    monkeypatch.delenv("CLOUDIQ_TEST_ROOT", raising=False)
    cfg = _write_config(
        tmp_path,
        """
        paths:
          smoke: "${CLOUDIQ_TEST_ROOT:-data}/_smoke_delta"
        """,
    )
    loader = ConfigLoader(cfg, env_path="nonexistent.env")
    assert loader.get("paths.smoke") == "data/_smoke_delta"


def test_env_interpolation(tmp_path: Path, monkeypatch) -> None:
    """${VAR} resolves from the environment when set."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    cfg = _write_config(
        tmp_path,
        """
        mlflow:
          tracking_uri: "${MLFLOW_TRACKING_URI:-mlruns}"
        """,
    )
    loader = ConfigLoader(cfg, env_path="nonexistent.env")
    assert loader.get("mlflow.tracking_uri") == "http://localhost:5000"


def test_validate_non_strict_passes_with_unresolved(tmp_path: Path) -> None:
    """validate(strict=False) returns True even with unresolved placeholders."""
    cfg = _write_config(
        tmp_path,
        """
        azure:
          storage_account: "${AZURE_STORAGE_ACCOUNT}"
        mlflow:
          tracking_uri: "${MLFLOW_TRACKING_URI:-mlruns}"
        """,
    )
    loader = ConfigLoader(cfg, env_path="nonexistent.env")
    assert loader.validate(strict=False) is True


def test_validate_strict_raises_with_unresolved(tmp_path: Path) -> None:
    """validate(strict=True) raises ValueError listing unresolved placeholders."""
    cfg = _write_config(
        tmp_path,
        """
        azure:
          storage_account: "${AZURE_STORAGE_ACCOUNT}"
          databricks:
            host: "${DATABRICKS_HOST}"
        """,
    )
    loader = ConfigLoader(cfg, env_path="nonexistent.env")
    with pytest.raises(ValueError) as excinfo:
        loader.validate(strict=True)
    message = str(excinfo.value)
    assert "AZURE_STORAGE_ACCOUNT" in message
    assert "DATABRICKS_HOST" in message


def test_validate_strict_passes_when_all_resolved(tmp_path: Path) -> None:
    """validate(strict=True) returns True when nothing is unresolved."""
    cfg = _write_config(
        tmp_path,
        """
        mlflow:
          tracking_uri: "${MLFLOW_TRACKING_URI:-mlruns}"
        """,
    )
    loader = ConfigLoader(cfg, env_path="nonexistent.env")
    assert loader.validate(strict=True) is True
