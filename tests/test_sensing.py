import math

import numpy as np
import pytest

import uav_search.envs.sensing as sensing
from uav_search.envs.sensing import bayes_update, binary_entropy, fov_offsets, profile_for_altitude


def test_profile_for_altitude_uses_nearest_reference_level():
    levels = [50.0, 100.0, 150.0]
    fov = [1, 5, 9]
    pd = [0.9, 0.8, 0.7]
    pf = [0.1, 0.2, 0.3]

    low = profile_for_altitude(55.0, levels, fov, pd, pf)
    mid = profile_for_altitude(92.0, levels, fov, pd, pf)
    high = profile_for_altitude(141.0, levels, fov, pd, pf)

    assert (low.level_index, low.reference_altitude_m, low.fov_cells, low.pd, low.pf) == (0, 50.0, 1, 0.9, 0.1)
    assert (mid.level_index, mid.reference_altitude_m, mid.fov_cells, mid.pd, mid.pf) == (1, 100.0, 5, 0.8, 0.2)
    assert (high.level_index, high.reference_altitude_m, high.fov_cells, high.pd, high.pf) == (2, 150.0, 9, 0.7, 0.3)


def test_continuous_profile_preserves_liu_anchor_probabilities_and_interpolates_between_them():
    levels = [50.0, 100.0, 150.0]
    pd = [0.9, 0.8, 0.7]
    pf = [0.1, 0.2, 0.3]

    low = sensing.continuous_profile_for_altitude(50.0, levels, pd, pf, full_fov_deg=90.0)
    mid = sensing.continuous_profile_for_altitude(100.0, levels, pd, pf, full_fov_deg=90.0)
    high = sensing.continuous_profile_for_altitude(150.0, levels, pd, pf, full_fov_deg=90.0)
    between = sensing.continuous_profile_for_altitude(75.0, levels, pd, pf, full_fov_deg=90.0)

    assert (low.pd, low.pf) == pytest.approx((0.9, 0.1))
    assert (mid.pd, mid.pf) == pytest.approx((0.8, 0.2))
    assert (high.pd, high.pf) == pytest.approx((0.7, 0.3))
    assert (between.pd, between.pf) == pytest.approx((0.85, 0.15))
    assert between.fov_radius_m == pytest.approx(75.0)


def test_continuous_fov_geometry_reproduces_liu_1_5_9_anchors_on_100m_grid():
    assert set(sensing.continuous_fov_offsets(50.0, 100.0)) == {(0, 0)}
    assert set(sensing.continuous_fov_offsets(100.0, 100.0)) == {
        (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)
    }
    assert set(sensing.continuous_fov_offsets(150.0, 100.0)) == {
        (dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
    }


def test_continuous_profile_rejects_altitudes_outside_calibration_range():
    with pytest.raises(ValueError, match="calibration range"):
        sensing.continuous_profile_for_altitude(
            49.0,
            [50.0, 100.0, 150.0],
            [0.9, 0.8, 0.7],
            [0.1, 0.2, 0.3],
            full_fov_deg=90.0,
        )


def test_fov_offsets_support_paper_sizes_1_5_9():
    assert fov_offsets(1) == ((0, 0),)
    assert set(fov_offsets(5)) == {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
    assert len(fov_offsets(9)) == 9
    assert set(fov_offsets(9)) == {(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)}


def test_bayes_update_matches_analytical_positive_and_negative_measurements():
    prior = 0.5
    pd = 0.8
    pf = 0.2

    positive = bayes_update(prior, True, pd, pf)
    negative = bayes_update(prior, False, pd, pf)

    assert np.isclose(positive, 0.8, rtol=0.0, atol=1e-12)
    assert np.isclose(negative, 0.2, rtol=0.0, atol=1e-12)


def test_binary_entropy_is_stable_at_probability_limits_and_maximal_at_half():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert math.isclose(binary_entropy(0.5), 1.0, rel_tol=0.0, abs_tol=1e-12)
