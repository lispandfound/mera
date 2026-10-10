"""Properties of the full REML fit (through the optimiser)."""

import numpy as np
import pytest
import scipy.optimize
from hypothesis import assume, given
from hypothesis import strategies as st
from strategies import identifiable_designs, residual_values, variances

from mera import decompose
from mera.core import _THETA_MAX, solve_mme


def fit_tol(r):
    """Tolerance for comparing two optimiser runs.

    Variance components near zero sit on a flat stretch of the likelihood.
    A std of 1e-5 and one of 1e-4 are both "zero" relative to the data, so the
    absolute tolerance scales with the data.
    """
    return {"rtol": 1e-5, "atol": 1e-5 * r.std()}


@given(identifiable_designs, st.tuples(*[st.floats(-0.5, 0.5)] * 3))
def test_fit_is_a_local_minimum(data, step):
    r, e, s = data
    fit = decompose(r, e, s)
    sds = np.array([fit.tau, fit.phi_s2s, fit.phi_ss])
    perturbed = np.maximum(np.abs(sds + np.array(step) * r.std()), 1e-3 * r.std())
    # Stay inside the optimiser's feasible region.
    assume(np.all(perturbed[:2] / perturbed[2] <= _THETA_MAX))
    other = solve_mme(r, e, s, *perturbed**2).neg2_reml
    assert fit.neg2_reml <= other + 1e-6 * (1 + abs(other))


@given(
    identifiable_designs,
    st.floats(0.1, 10) | st.floats(-10, -0.1),
    st.floats(-1e6, 1e6),
    st.randoms(use_true_random=False),
)
def test_affine_map_permutation_and_relabelling(data, k, c, rnd):
    """Fitting k·r + c, rows shuffled and ids relabelled, maps every output."""
    r, e, s = data
    perm = np.array(rnd.sample(range(len(r)), len(r)))
    # Relabel so the sorted order of events and sites changes too.
    ev_label, st_label = (
        np.array([f"ev{-i}" for i in e]),
        np.array([f"st{-i}" for i in s]),
    )
    base = decompose(r, e, s)
    other = decompose(k * r[perm] + c, ev_label[perm], st_label[perm])
    ev = np.searchsorted(other.events, [f"ev{-i}" for i in base.events])
    st_ = np.searchsorted(other.sites, [f"st{-i}" for i in base.sites])
    tol = fit_tol(r)
    np.testing.assert_allclose((other.bias - c) / k, base.bias, **tol)
    for name in ("remaining", "within_event"):
        np.testing.assert_allclose(
            getattr(other, name) / k, getattr(base, name)[perm], **tol
        )
    for name in ("bias_std", "tau", "phi_s2s", "phi_ss"):
        np.testing.assert_allclose(
            getattr(other, name) / abs(k), getattr(base, name), **tol
        )
    np.testing.assert_allclose(other.event_terms[ev] / k, base.event_terms, **tol)
    np.testing.assert_allclose(other.site_terms[st_] / k, base.site_terms, **tol)
    np.testing.assert_allclose(other.event_std[ev] / abs(k), base.event_std, **tol)
    np.testing.assert_allclose(other.site_std[st_] / abs(k), base.site_std, **tol)


@given(
    st.lists(
        st.tuples(st.integers(0, 3), st.integers(0, 3), residual_values),
        min_size=1,
        max_size=20,
    ),
    st.booleans(),
)
def test_any_input_is_rejected_or_finite(records, constant):
    e, s, r = (np.array(x) for x in zip(*records))
    if constant:
        r = np.full_like(r, r[0])
    try:
        fit = decompose(r, e, s)
    except ValueError:
        no_variance = np.ptp(r) == 0 or not r.std() > 0
        assert len(np.unique(e)) < 2 or len(np.unique(s)) < 2 or no_variance
        return
    except RuntimeError:  # φ_SS not identifiable, as in tiny designs that fit exactly
        return
    outputs = np.r_[
        fit.bias,
        fit.bias_std,
        fit.tau,
        fit.phi_s2s,
        fit.phi_ss,
        fit.neg2_reml,
        fit.event_terms,
        fit.event_std,
        fit.site_terms,
        fit.site_std,
        fit.remaining,
    ]
    assert np.all(np.isfinite(outputs))


VALID = ([1.0, 2.0, 0.5, 1.5], [0, 0, 1, 1], [0, 1, 0, 1])


@pytest.mark.parametrize(
    "r, e, s, initial",
    [
        ([1.0, np.nan, 2.0], [0, 1, 1], [0, 1, 0], None),  # missing residual
        ([1.0, 2.0], [0, 1], [0, 1, 1], None),  # length mismatch
        ([1.0, 1.0, 1.0, 1.0], VALID[1], VALID[2], None),  # constant residuals
        ([1.0, 2.0, 3.0], [0, 0, 0], [0, 1, 2], None),  # one event
        (VALID[0], [0.0, 0.0, np.nan, np.nan], VALID[2], None),  # missing event id
        (VALID[0], VALID[1], [0.0, np.nan, 0.0, np.nan], None),  # missing site id
        (*VALID, (0.1, 0.1, 0.0)),  # φ_SS² = 0
        (*VALID, (-0.1, 0.1, 0.1)),  # negative τ²
        (*VALID, (0.1, np.nan, 0.1)),  # NaN φ_S2S²
        (*VALID, (0.1, np.inf, 0.1)),  # infinite φ_S2S²
    ],
)
def test_invalid_input_raises(r, e, s, initial):
    with pytest.raises(ValueError):
        decompose(np.array(r), np.array(e), np.array(s), initial)


@pytest.mark.parametrize(
    "e, s, variances",
    [
        ([0, 0, 2, 2], VALID[2], (1, 1, 1)),  # event code 1 missing
        ([0, 0, -1, -1], VALID[2], (1, 1, 1)),  # negative code
        ([0.0, 0.0, 1.0, 1.0], VALID[2], (1, 1, 1)),  # non-integer codes
        (VALID[1], VALID[2], (1, 1, 0)),  # φ_SS² = 0
        (VALID[1], VALID[2], (-1, 1, 1)),  # negative τ²
    ],
)
def test_solve_mme_rejects_invalid_input(e, s, variances):
    with pytest.raises(ValueError):
        solve_mme(np.array(VALID[0]), np.array(e), np.array(s), *variances)


def test_noise_free_remaining_residual_raises():
    """Near-noise-free data put τ/φ_SS past the cap, which must raise, not bias the fit."""
    rng = np.random.default_rng(1)
    e, s = np.nonzero(rng.random((30, 40)) < 0.5)
    r = rng.normal(size=30)[e] + rng.normal(size=40)[s] + rng.normal(0, 1e-4, len(e))
    with pytest.raises(RuntimeError, match="not identifiable"):
        decompose(r, e, s)
    # With the noise 100 times larger the ratio is ~100, inside the cap.
    r = r + rng.normal(0, 1e-2, len(e))
    fit = decompose(r, e, s)
    assert 0.005 < fit.phi_ss < 0.02


@st.composite
def balanced_grids(draw):
    """Complete event × site grid, one record per pair, built from random effects."""
    E, S = draw(st.integers(3, 6)), draw(st.integers(3, 6))
    floats = lambda n: np.array(draw(st.lists(residual_values, min_size=n, max_size=n)))
    y = (
        floats(1)[0]
        + floats(E)[:, None]
        + floats(S)[None, :]
        + floats(E * S).reshape(E, S)
    )
    return y


@given(balanced_grids())
def test_balanced_grid_matches_anova(y):
    """For balanced crossed data, REML equals the positive ANOVA estimators."""
    E, S = y.shape
    grand, ev_mean, st_mean = y.mean(), y.mean(axis=1), y.mean(axis=0)
    ms_event = S * np.sum((ev_mean - grand) ** 2) / (E - 1)
    ms_site = E * np.sum((st_mean - grand) ** 2) / (S - 1)
    ms_res = np.sum((y - ev_mean[:, None] - st_mean[None, :] + grand) ** 2) / (
        (E - 1) * (S - 1)
    )
    expected = np.array([(ms_event - ms_res) / S, (ms_site - ms_res) / E, ms_res])
    # Keep away from the boundary, where REML truncates and ANOVA goes negative.
    assume(np.all(expected > 0.05 * y.var()))

    e, s = np.indices(y.shape)
    fit = decompose(y.ravel(), e.ravel(), s.ravel())
    np.testing.assert_allclose(
        [fit.tau**2, fit.phi_s2s**2, fit.phi_ss**2], expected, rtol=1e-5
    )
    np.testing.assert_allclose(fit.bias, grand, atol=1e-9)
    # Balanced BLUPs equal the marginal means times a shrinkage factor.
    tau2, phi_s2s2, phi_ss2 = expected
    np.testing.assert_allclose(
        fit.event_terms,
        S * tau2 / (S * tau2 + phi_ss2) * (ev_mean - grand),
        rtol=1e-4,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        fit.site_terms,
        E * phi_s2s2 / (E * phi_s2s2 + phi_ss2) * (st_mean - grand),
        rtol=1e-4,
        atol=1e-6,
    )


@given(identifiable_designs, variances)
def test_warm_start_reaches_the_same_fit(data, start):
    r, e, s = data
    base = decompose(r, e, s)
    warm = decompose(r, e, s, initial_variances=start)
    np.testing.assert_allclose(warm.neg2_reml, base.neg2_reml, rtol=1e-8, atol=1e-8)
    for name in ("bias", "tau", "phi_s2s", "phi_ss", "phi", "sigma"):
        np.testing.assert_allclose(
            getattr(warm, name), getattr(base, name), **fit_tol(r)
        )


def test_optimiser_failure_raises(monkeypatch):
    minimize = scipy.optimize.minimize

    def failing(*args, **kwargs):
        result = minimize(*args, **kwargs)
        result.success, result.message = False, "forced failure"
        return result

    monkeypatch.setattr(scipy.optimize, "minimize", failing)
    rng = np.random.default_rng(0)
    e, s = rng.integers(0, 5, 50), rng.integers(0, 5, 50)
    with pytest.raises(RuntimeError, match="forced failure"):
        decompose(rng.normal(size=50), e, s)
