"""Tests for the validated boundary: Contract and Market."""

from __future__ import annotations

import dataclasses

import pytest

from bsm.types import Contract, Market, OptionKind


class TestOptionKind:
    def test_string_membership(self) -> None:
        assert OptionKind("call") is OptionKind.CALL
        assert OptionKind("put") is OptionKind.PUT

    def test_compares_equal_to_string(self) -> None:
        """StrEnum: usable directly as a CLI argument or database value."""
        kind: str = OptionKind.CALL
        assert kind == "call"

    def test_unknown_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="not a valid OptionKind"):
            OptionKind("straddle")


class TestContract:
    def test_construction(self) -> None:
        contract = Contract(strike=100.0, maturity=0.5, kind=OptionKind.CALL)
        assert contract.strike == 100.0
        assert contract.maturity == 0.5
        assert not contract.is_expired

    def test_zero_maturity_is_expired(self) -> None:
        assert Contract(100.0, 0.0, OptionKind.CALL).is_expired

    @pytest.mark.parametrize("strike", [0.0, -1.0])
    def test_non_positive_strike_rejected(self, strike: float) -> None:
        with pytest.raises(ValueError, match="strike must be strictly positive"):
            Contract(strike, 1.0, OptionKind.CALL)

    def test_negative_maturity_rejected(self) -> None:
        with pytest.raises(ValueError, match="maturity must be non-negative"):
            Contract(100.0, -0.5, OptionKind.CALL)

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            Contract(bad, 1.0, OptionKind.CALL)
        with pytest.raises(ValueError, match="must be finite"):
            Contract(100.0, bad, OptionKind.CALL)

    def test_frozen(self) -> None:
        contract = Contract(100.0, 1.0, OptionKind.CALL)
        with pytest.raises(dataclasses.FrozenInstanceError):
            contract.strike = 110.0  # type: ignore[misc]

    def test_hashable(self) -> None:
        """Frozen and slotted, so it can key a cache or a scenario log."""
        first = Contract(100.0, 1.0, OptionKind.CALL)
        second = Contract(100.0, 1.0, OptionKind.CALL)
        assert {first, second} == {first}

    def test_slotted_has_no_instance_dict(self) -> None:
        """slots=True: no per-instance __dict__, so a typo cannot silently attach."""
        assert not hasattr(Contract(100.0, 1.0, OptionKind.CALL), "__dict__")


class TestMarket:
    def test_construction_with_default_yield(self) -> None:
        market = Market(spot=100.0, rate=0.05, volatility=0.2)
        assert market.dividend_yield == 0.0

    @pytest.mark.parametrize("spot", [0.0, -10.0])
    def test_non_positive_spot_rejected(self, spot: float) -> None:
        with pytest.raises(ValueError, match="spot must be strictly positive"):
            Market(spot, 0.05, 0.2)

    def test_negative_volatility_rejected(self) -> None:
        with pytest.raises(ValueError, match="volatility must be non-negative"):
            Market(100.0, 0.05, -0.2)

    def test_zero_volatility_allowed(self) -> None:
        """Admissible: it yields the discounted intrinsic value."""
        assert Market(100.0, 0.05, 0.0).volatility == 0.0

    def test_negative_rate_allowed(self) -> None:
        """Several sovereign curves have been negative; the model does not object."""
        assert Market(100.0, -0.01, 0.2).rate == -0.01

    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="must be finite"):
            Market(100.0, bad, 0.2)

    def test_frozen(self) -> None:
        market = Market(100.0, 0.05, 0.2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            market.spot = 110.0  # type: ignore[misc]
