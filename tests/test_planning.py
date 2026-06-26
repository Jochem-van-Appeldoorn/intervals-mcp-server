"""
Unit tests for planning helpers in intervals_mcp_server.tools.planning.

Covers:
- _assess_basis: all scenarios (present, absent, no data, no FTP)
- _extract_30d_power: valid response, empty response, missing durations
- _determine_phases with basis_present=True/False/None
"""

import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
os.environ.setdefault("API_KEY", "test")
os.environ.setdefault("ATHLETE_ID", "i1")

from intervals_mcp_server.tools.planning import (
    _assess_basis,
    _extract_30d_power,
    _determine_phases,
    _BASIS_20MIN_THRESHOLD,
    _BASIS_60MIN_THRESHOLD,
)


# ---------------------------------------------------------------------------
# _assess_basis
# ---------------------------------------------------------------------------

class TestAssessBasis:
    def test_basis_present_with_ftp(self):
        ftp = 300
        power_20 = ftp * _BASIS_20MIN_THRESHOLD  # exactly at threshold
        power_60 = ftp * _BASIS_60MIN_THRESHOLD
        basis, ftp_ref, r20, r60 = _assess_basis(power_20, power_60, ftp)
        assert basis is True
        assert ftp_ref == float(ftp)
        assert abs(r20 - _BASIS_20MIN_THRESHOLD) < 0.001
        assert abs(r60 - _BASIS_60MIN_THRESHOLD) < 0.001

    def test_basis_absent_weak_20min(self):
        ftp = 300
        power_20 = ftp * (_BASIS_20MIN_THRESHOLD - 0.05)  # below threshold
        power_60 = ftp * _BASIS_60MIN_THRESHOLD
        basis, _, r20, _ = _assess_basis(power_20, power_60, ftp)
        assert basis is False
        assert r20 < _BASIS_20MIN_THRESHOLD

    def test_basis_absent_weak_60min(self):
        ftp = 300
        power_20 = ftp * _BASIS_20MIN_THRESHOLD
        power_60 = ftp * (_BASIS_60MIN_THRESHOLD - 0.05)  # below threshold
        basis, _, _, r60 = _assess_basis(power_20, power_60, ftp)
        assert basis is False
        assert r60 < _BASIS_60MIN_THRESHOLD

    def test_no_data_returns_none(self):
        basis, ftp_ref, r20, r60 = _assess_basis(None, None, 300)
        assert basis is None
        assert ftp_ref is None
        assert r20 is None
        assert r60 is None

    def test_ftp_proxy_from_20min(self):
        """When FTP is not given, use 20-min power × 0.95 as proxy."""
        power_20 = 300.0
        # 60-min must be >= proxy * threshold = 285 * 0.75 = 213.75
        power_60 = 220.0
        basis, ftp_ref, r20, r60 = _assess_basis(power_20, power_60, ftp=None)
        assert ftp_ref == pytest.approx(power_20 * 0.95)
        # 20-min ratio: 300 / 285 ≈ 1.053 → above threshold
        assert r20 is not None and r20 > _BASIS_20MIN_THRESHOLD
        assert basis is True

    def test_ftp_proxy_basis_absent(self):
        """Proxy FTP: 20-min power is by definition 5% above proxy, but 60-min may fail."""
        power_20 = 300.0
        power_60 = 100.0  # far below threshold
        basis, _, _, r60 = _assess_basis(power_20, power_60, ftp=None)
        assert basis is False
        assert r60 < _BASIS_60MIN_THRESHOLD

    def test_only_20min_available(self):
        """60-min missing → cannot fully confirm basis."""
        basis, _, r20, r60 = _assess_basis(280.0, None, ftp=300)
        assert basis is False  # meets_60 is False because r60 is None
        assert r20 is not None
        assert r60 is None

    def test_only_60min_no_ftp(self):
        """No 20-min and no FTP → cannot derive ftp_ref → None."""
        basis, ftp_ref, _, _ = _assess_basis(None, 240.0, ftp=None)
        assert basis is None
        assert ftp_ref is None


# ---------------------------------------------------------------------------
# _extract_30d_power
# ---------------------------------------------------------------------------

def _make_curve_response(secs: list[int], values: list[float | None]) -> dict:
    return {"list": [{"secs": secs, "values": values}]}


class TestExtract30dPower:
    def test_extracts_20min_and_60min(self):
        secs = [60, 300, 1200, 3600]
        values = [500.0, 420.0, 320.0, 280.0]
        p20, p60 = _extract_30d_power(_make_curve_response(secs, values))
        assert p20 == pytest.approx(320.0)
        assert p60 == pytest.approx(280.0)

    def test_missing_60min_returns_none(self):
        secs = [60, 1200]
        values = [500.0, 320.0]
        p20, p60 = _extract_30d_power(_make_curve_response(secs, values))
        assert p20 == pytest.approx(320.0)
        assert p60 is None

    def test_empty_list_returns_none(self):
        p20, p60 = _extract_30d_power({"list": []})
        assert p20 is None
        assert p60 is None

    def test_error_response_returns_none(self):
        p20, p60 = _extract_30d_power({"error": "Not found"})
        assert p20 is None
        assert p60 is None

    def test_non_dict_returns_none(self):
        p20, p60 = _extract_30d_power(None)
        assert p20 is None
        assert p60 is None

    def test_null_value_returns_none(self):
        secs = [1200, 3600]
        values = [None, 280.0]
        p20, p60 = _extract_30d_power(_make_curve_response(secs, values))
        assert p20 is None
        assert p60 == pytest.approx(280.0)


# ---------------------------------------------------------------------------
# _determine_phases with basis_present
# ---------------------------------------------------------------------------

import pytest


class TestDeterminePhasesBasisPresent:
    """basis_present=True: skip or shorten Base phase."""

    def test_basis_present_high_ratio_goes_to_build(self):
        # ratio >= 0.70 + basis present → Build only (no Base)
        phases = _determine_phases(20, current_ctl=80, goal_ctl=100, basis_present=True)
        names = [p for p, _ in phases]
        assert "base" not in names
        assert "build" in names

    def test_basis_present_low_ratio_still_has_base(self):
        # ratio < 0.70 + basis present → short Base then Build
        phases = _determine_phases(20, current_ctl=50, goal_ctl=120, basis_present=True)
        names = [p for p, _ in phases]
        assert "base" in names
        assert "build" in names
        # Base should be shorter than standard (≤ remaining // 3 equiv)
        base_wks = next(w for p, w in phases if p == "base")
        build_wks = next(w for p, w in phases if p == "build")
        # Short base: base_wks <= build_wks (build dominates)
        assert build_wks >= base_wks

    def test_basis_present_no_preparation_phase(self):
        # Even with low ratio, basis_present skips Preparation
        phases = _determine_phases(30, current_ctl=40, goal_ctl=120, basis_present=True)
        names = [p for p, _ in phases]
        assert "preparation" not in names


class TestDeterminePhasesBasisAbsent:
    """basis_present=False: force Base even when CTL ratio is high."""

    def test_basis_absent_high_ratio_adds_base(self):
        # ratio >= 0.90 but no recent riding → should add Base
        phases = _determine_phases(20, current_ctl=95, goal_ctl=100, basis_present=False)
        names = [p for p, _ in phases]
        assert "base" in names

    def test_basis_absent_moderate_ratio_adds_base(self):
        # ratio 70–90% + basis absent → Base + Build
        phases = _determine_phases(20, current_ctl=75, goal_ctl=100, basis_present=False)
        names = [p for p, _ in phases]
        assert "base" in names
        assert "build" in names

    def test_basis_absent_low_ratio_full_plan(self):
        # ratio < 0.70 + basis absent → Prep + Base + Build
        phases = _determine_phases(30, current_ctl=40, goal_ctl=120, basis_present=False)
        names = [p for p, _ in phases]
        assert "preparation" in names
        assert "base" in names
        assert "build" in names


class TestDeterminePhasesBasisNone:
    """basis_present=None: original CTL-gap logic, unchanged."""

    def test_none_high_ratio_straight_to_build(self):
        phases = _determine_phases(20, current_ctl=95, goal_ctl=100, basis_present=None)
        names = [p for p, _ in phases]
        assert "base" not in names
        assert "build" in names

    def test_none_moderate_ratio_base_plus_build(self):
        phases = _determine_phases(20, current_ctl=75, goal_ctl=100, basis_present=None)
        names = [p for p, _ in phases]
        assert "base" in names
        assert "build" in names

    def test_none_low_ratio_full_plan(self):
        phases = _determine_phases(30, current_ctl=40, goal_ctl=120, basis_present=None)
        names = [p for p, _ in phases]
        assert "preparation" in names
        assert "base" in names
        assert "build" in names

    def test_none_is_default_parameter(self):
        """Omitting basis_present gives same result as None."""
        phases_explicit = _determine_phases(20, 75.0, 100.0, basis_present=None)
        phases_default = _determine_phases(20, 75.0, 100.0)
        assert phases_explicit == phases_default


class TestDeterminePhasesEdgeCases:
    def test_short_plan_ignores_basis(self):
        # remaining <= 4: always collapse to build, ignore basis signal
        phases_t = _determine_phases(7, 50.0, 120.0, basis_present=True)
        phases_f = _determine_phases(7, 50.0, 120.0, basis_present=False)
        # Both should have build (not base+build) since remaining is small
        # peak=2, race=1 → remaining=4
        names_t = [p for p, _ in phases_t]
        names_f = [p for p, _ in phases_f]
        assert "build" in names_t
        assert "build" in names_f

    def test_peak_and_race_always_present(self):
        for basis in (True, False, None):
            phases = _determine_phases(20, 80.0, 100.0, basis_present=basis)
            names = [p for p, _ in phases]
            assert "peak" in names
            assert "race" in names

    def test_total_weeks_preserved(self):
        """Sum of all phase weeks must equal total_weeks."""
        for basis in (True, False, None):
            for total in (8, 16, 24, 32):
                phases = _determine_phases(total, 60.0, 120.0, basis_present=basis)
                assert sum(w for _, w in phases) == total
