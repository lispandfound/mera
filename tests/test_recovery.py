"""Statistical recovery of known parameters from simulated data (seeded, not Hypothesis)."""

import time

import numpy as np
import pytest

from mera import decompose

TRUE = {"bias": 0.1, "tau": 0.4, "phi_s2s": 0.35, "phi_ss": 0.5}


def simulate(rng, n_events, n_sites, p_record):
    """Unbalanced crossed design: each event recorded at a random subset of sites."""
    keep = rng.random((n_events, n_sites)) < p_record
    e, s = np.nonzero(keep)
    dB = rng.normal(0, TRUE["tau"], n_events)
    dS2S = rng.normal(0, TRUE["phi_s2s"], n_sites)
    r = TRUE["bias"] + dB[e] + dS2S[s] + rng.normal(0, TRUE["phi_ss"], len(e))
    return r, e, s, dB, dS2S


def test_recovers_parameters_and_calibrated_uncertainty():
    rng = np.random.default_rng(20261010)
    fits = [
        (decompose(r, e, s), dB, dS2S)
        for r, e, s, dB, dS2S in (simulate(rng, 40, 60, 0.3) for _ in range(100))
    ]

    # REML variance estimates are (near) unbiased: the mean over simulations
    # lies within 4 standard errors of the truth.
    for name in ("tau", "phi_s2s", "phi_ss"):
        est = np.array([getattr(f, name) ** 2 for f, _, _ in fits])
        se = est.std(ddof=1) / np.sqrt(len(est))
        assert abs(est.mean() - TRUE[name] ** 2) < 4 * se, name
    bias = np.array([f.bias for f, _, _ in fits])
    assert abs(bias.mean() - TRUE["bias"]) < 4 * bias.std(ddof=1) / np.sqrt(len(bias))

    # Random terms track the truth, and ±1.96 conditional std covers ~95%.
    for terms, std, true in (
        ("event_terms", "event_std", 1),
        ("site_terms", "site_std", 2),
    ):
        corr = np.mean([np.corrcoef(getattr(f[0], terms), f[true])[0, 1] for f in fits])
        assert corr > 0.8, terms
        z = np.concatenate(
            [(getattr(f[0], terms) - f[true]) / getattr(f[0], std) for f in fits]
        )
        assert 0.93 < np.mean(np.abs(z) < 1.96) < 0.97, terms


@pytest.mark.slow
def test_scales_to_tens_of_thousands_of_sites():
    """Complete simulation grid as in sim-vs-sim comparisons (~1.5M records, ~0.3 GB)."""
    rng = np.random.default_rng(1)
    r, e, s, _, dS2S = simulate(rng, 150, 10_000, 1.0)
    start = time.perf_counter()
    fit = decompose(r, e, s)
    elapsed = time.perf_counter() - start
    np.testing.assert_allclose(
        [fit.tau, fit.phi_s2s, fit.phi_ss],
        [TRUE[k] for k in ("tau", "phi_s2s", "phi_ss")],
        rtol=0.15,
    )
    assert np.corrcoef(fit.site_terms, dS2S)[0, 1] > 0.95
    assert elapsed < 300
