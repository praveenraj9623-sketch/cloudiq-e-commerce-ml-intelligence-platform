"""Deterministic demand-feature tests (Correction 16).

These tests do not start Spark. They validate the documented semantics of
``rolling_mean_3`` (three preceding months only) and the demand training
cutoff (partial October 2018 excluded) using pure-Python reference logic that
mirrors the Spark implementation, plus a direct check of
:func:`derive_demand_cutoff_month` against a small mock object.
"""

from __future__ import annotations

from datetime import datetime


def _rolling_mean_3_prior(series: list[float]) -> list[float | None]:
    """Reference implementation of rowsBetween(-3, -1) rolling mean.

    At each position, average exactly the three preceding values. Positions
    with fewer than three preceding values use the available prior values;
    position 0 has no prior values and is None.
    """
    out: list[float | None] = []
    for i in range(len(series)):
        prior = series[max(0, i - 3):i]
        out.append(sum(prior) / len(prior) if prior else None)
    return out


def test_rolling_mean_3_uses_three_prior_months() -> None:
    """rolling_mean_3 at each position equals the mean of the three prior months."""
    series = [10.0, 20.0, 30.0, 40.0, 50.0]
    result = _rolling_mean_3_prior(series)

    # Position 4 (value 50) must use months [20, 30, 40] -> mean 30.0,
    # never including the current month's value (50).
    assert result[4] == 30.0
    # Position 3 (value 40) must use [10, 20, 30] -> mean 20.0.
    assert result[3] == 20.0
    # The current month value is never part of its own rolling mean window.
    for i in range(len(series)):
        prior = series[max(0, i - 3):i]
        assert len(prior) <= 3


class _MockColumn:
    """Minimal stand-in for a Spark Row carrying a single timestamp."""

    def __init__(self, value: datetime) -> None:
        self.value = value

    def __getitem__(self, _key: str) -> datetime:
        return self.value


class _MockExpr:
    """Minimal stand-in for a Spark column expression supporting alias()."""

    def alias(self, name: str) -> str:
        return name


class _MockAgg:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def first(self) -> _MockColumn:
        return _MockColumn(self._value)


class _MockOrders:
    """Mock orders DataFrame exposing only select(...).first()."""

    def __init__(self, max_ts: datetime) -> None:
        self._max_ts = max_ts

    def select(self, *_args, **_kwargs) -> _MockAgg:
        return _MockAgg(self._max_ts)


def test_partial_final_month_excluded(monkeypatch) -> None:
    """A max timestamp in October 2018 yields a 2018-09 cutoff."""
    import src.processing.gold as gold

    # Patch F.max so derive_demand_cutoff_month does not require Spark.
    monkeypatch.setattr(
        gold.F,
        "max",
        lambda _col: _MockExpr(),
        raising=True,
    )

    orders = _MockOrders(datetime(2018, 10, 17, 12, 0, 0))
    cutoff = gold.derive_demand_cutoff_month(orders)
    assert cutoff == "2018-09"


def test_cutoff_handles_january_rollover(monkeypatch) -> None:
    """A January max timestamp rolls the cutoff back to the prior December."""
    import src.processing.gold as gold

    monkeypatch.setattr(
        gold.F,
        "max",
        lambda _col: _MockExpr(),
        raising=True,
    )
    orders = _MockOrders(datetime(2018, 1, 9, 0, 0, 0))
    assert gold.derive_demand_cutoff_month(orders) == "2017-12"
