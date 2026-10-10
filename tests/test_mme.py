"""Exact properties of the mixed-model equations at fixed variances.

These hold to floating-point precision for any data and any variances, so
they test the solver independently of the REML optimiser.
"""

import numpy as np
from hypothesis import given
from strategies import designs, variances

from mera.core import solve_mme


def dense_reference(r, e, s, tau2, phi_s2s2, phi_ss2):
    """Textbook formulas with the full N×N covariance V."""
    N, E, S = len(r), e.max() + 1, s.max() + 1
    X = np.ones((N, 1))
    Ze, Zs = np.eye(E)[e], np.eye(S)[s]
    V = tau2 * Ze @ Ze.T + phi_s2s2 * Zs @ Zs.T + phi_ss2 * np.eye(N)
    V_inv = np.linalg.inv(V)
    XtViX = X.T @ V_inv @ X
    bias = np.linalg.solve(XtViX, X.T @ V_inv @ r)[0]
    res = r - bias
    neg2 = np.linalg.slogdet(V)[1] + np.linalg.slogdet(XtViX)[1] + res @ V_inv @ res
    # Prediction-error covariance: inverse of Henderson's coefficient matrix.
    T = np.hstack([X, Ze, Zs])
    G_inv = np.diag(np.r_[0.0, np.full(E, 1 / tau2), np.full(S, 1 / phi_s2s2)])
    std = np.sqrt(np.diag(np.linalg.inv(T.T @ T / phi_ss2 + G_inv)))
    event_terms = tau2 * Ze.T @ V_inv @ res
    site_terms = phi_s2s2 * Zs.T @ V_inv @ res
    return {
        "bias": bias,
        "event_terms": event_terms,
        "site_terms": site_terms,
        "within_event": res - event_terms[e],
        "remaining": res - event_terms[e] - site_terms[s],
        "neg2_reml": neg2,
        "bias_std": std[0],
        "event_std": std[1 : 1 + E],
        "site_std": std[1 + E :],
    }


@given(designs(), variances)
def test_matches_dense_reference(data, var):
    r, e, s = data
    sol = solve_mme(r, e, s, *var)
    ref = dense_reference(r, e, s, *var)
    for name, expected in ref.items():
        np.testing.assert_allclose(
            getattr(sol, name), expected, rtol=1e-7, atol=1e-9, err_msg=name
        )
