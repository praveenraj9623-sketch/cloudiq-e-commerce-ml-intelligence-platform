"""Configuration loading for CloudIQ.

Loads ``config.yaml``, resolves ``${VAR}`` and ``${VAR:-default}`` environment
placeholders recursively, and exposes dot-notation access plus path helpers.

Validation is local-first (Correction 7): :meth:`ConfigLoader.validate` defaults
to ``strict=False``, logging a warning per unresolved placeholder and returning
``True`` so local runs succeed without Azure/Databricks variables. Cloud
deployment contexts call ``validate(strict=True)`` to raise on any unresolved
placeholder.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.utils.logger import get_logger

# Matches ${VAR} and ${VAR:-default}. The default group is optional.
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")
# Matches any remaining unresolved ${...} after resolution.
_UNRESOLVED_RE = re.compile(r"\$\{[^}]*\}")


class ConfigLoader:
    """Load and resolve the CloudIQ YAML configuration.

    Args:
        config_path: Path to the YAML config file.
        env_path: Path to a dotenv file loaded before resolution.
    """

    def __init__(
        self,
        config_path: str = "config.yaml",
        env_path: str = ".env",
    ) -> None:
        self.logger = get_logger("utils.config")
        self.config_path = Path(config_path)

        if Path(env_path).exists():
            load_dotenv(env_path)
            self.logger.debug("Loaded environment from {}", env_path)

        with self.config_path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        self._config: dict[str, Any] = self._resolve_placeholders(raw)
        self.logger.debug("Loaded config from {}", config_path)

    def _resolve_string(self, value: str) -> str:
        """Resolve ``${VAR}`` / ``${VAR:-default}`` patterns within a string.

        Unset variables without a default are left untouched so that
        :meth:`validate` can detect them.
        """

        def _replace(match: re.Match[str]) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value not in (None, ""):
                return env_value
            if default is not None:
                return default
            # Leave unresolved so validate() can report it.
            return match.group(0)

        return _PLACEHOLDER_RE.sub(_replace, value)

    def _resolve_placeholders(self, obj: Any) -> Any:
        """Recursively resolve placeholders in strings, dicts, and lists."""
        if isinstance(obj, str):
            return self._resolve_string(obj)
        if isinstance(obj, dict):
            return {k: self._resolve_placeholders(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._resolve_placeholders(v) for v in obj]
        return obj

    def get(self, key_path: str, default: Any = None) -> Any:
        """Return a value by dot-notation key path.

        Args:
            key_path: e.g. ``"models.churn.churn_days_threshold"``.
            default: Returned if any key in the path is missing.

        Returns:
            The resolved value, or ``default`` if not found.
        """
        node: Any = self._config
        for part in key_path.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def get_path(self, key_path: str, create: bool = True) -> Path:
        """Return a filesystem :class:`Path` for a configured path key.

        Args:
            key_path: Dot-notation key resolving to a path string.
            create: When ``True``, create the directory if it does not exist.

        Returns:
            The resolved :class:`Path`.

        Raises:
            KeyError: If the key path does not resolve to a value.
        """
        value = self.get(key_path)
        if value is None:
            raise KeyError(f"No path configured for '{key_path}'")
        path = Path(str(value))
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _find_unresolved(self) -> list[str]:
        """Return dot-notation key paths whose values still contain ``${...}``."""
        unresolved: list[str] = []

        def _walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, val in node.items():
                    _walk(val, f"{prefix}.{key}" if prefix else str(key))
            elif isinstance(node, list):
                for idx, val in enumerate(node):
                    _walk(val, f"{prefix}[{idx}]")
            elif isinstance(node, str) and _UNRESOLVED_RE.search(node):
                unresolved.append(f"{prefix}={node}")

        _walk(self._config, "")
        return unresolved

    def validate(self, strict: bool = False) -> bool:
        """Validate that placeholders are resolved (Correction 7).

        Args:
            strict: When ``False`` (default, local runs), log a warning for
                each unresolved placeholder and return ``True``. When ``True``
                (cloud deploy), raise :class:`ValueError` listing all
                unresolved placeholders.

        Returns:
            ``True`` when validation passes.

        Raises:
            ValueError: If ``strict`` is ``True`` and any placeholder is
                unresolved.
        """
        unresolved = self._find_unresolved()
        if not unresolved:
            return True

        if strict:
            raise ValueError(
                "Unresolved configuration placeholders: "
                + "; ".join(unresolved)
            )

        for item in unresolved:
            self.logger.warning("Unresolved placeholder (optional): {}", item)
        return True
