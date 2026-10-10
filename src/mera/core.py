"""Mixed-effects decomposition of ground-motion residuals.

The model, in the notation of Al Atik et al. (2010), fitted by restricted
maximum likelihood (REML):

    r_es = a + δB_e + δS2S_s + δWS_es

    δB_e    ~ N(0, τ²)        between-event term
    δS2S_s  ~ N(0, φ_S2S²)    site-to-site term
    δWS_es  ~ N(0, φ_SS²)     remaining (single-station within-event) residual

with a the fixed intercept (model bias) and δW_es = δS2S_s + δWS_es the
within-event residual. Events and sites form crossed random effects.

The solver uses Henderson's mixed-model equations in the relative-std
parameterisation of lme4 (Bates et al. 2015): θ_e = τ/φ_SS, θ_s = φ_S2S/φ_SS.
A closed form gives φ_SS² at any θ, so REML reduces to a 2-parameter
optimisation, and the fit can set a variance component to exactly zero.

The solver eliminates the larger grouping analytically. Its block of the
equations is diagonal, because each record has exactly one event and one site.
The Schur complement leaves a dense system of size 1 + min(#events, #sites),
and no N×N matrix ever exists.
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import scipy.linalg
import scipy.optimize

# Upper bound on the relative standard deviations τ/φ_SS and φ_S2S/φ_SS.
_THETA_MAX = 1e3


@dataclass(frozen=True)
class Decomposition:
    """Result of `decompose`.

    Per-record arrays follow the input order.

    Parameters
    ----------
    bias : float
        The fixed intercept a.
    bias_std : float
        Standard error of the bias.
    tau : float
        Between-event standard deviation τ.
    phi_s2s : float
        Site-to-site standard deviation φ_S2S.
    phi_ss : float
        Remaining (single-station within-event) standard deviation φ_SS.
    events : np.ndarray
        Unique event ids, in the order of `event_terms` and `event_std`.
    event_terms : np.ndarray
        Between-event terms δB_e.
    event_std : np.ndarray
        Conditional standard deviation of each δB_e.
    sites : np.ndarray
        Unique site ids, in the order of `site_terms` and `site_std`.
    site_terms : np.ndarray
        Site-to-site terms δS2S_s.
    site_std : np.ndarray
        Conditional standard deviation of each δS2S_s.
    within_event : np.ndarray
        Within-event residual δW_es = δS2S_s + δWS_es of each record.
    remaining : np.ndarray
        Remaining residual δWS_es of each record.
    neg2_reml : float
        −2 × the REML log-likelihood, up to an additive constant.
    """

    bias: float
    bias_std: float
    tau: float
    phi_s2s: float
    phi_ss: float
    events: np.ndarray
    event_terms: np.ndarray
    event_std: np.ndarray
    sites: np.ndarray
    site_terms: np.ndarray
    site_std: np.ndarray
    within_event: np.ndarray
    remaining: np.ndarray
    neg2_reml: float

    @property
    def phi(self) -> float:
        """float: Within-event standard deviation φ = √(φ_S2S² + φ_SS²)."""
        return math.hypot(self.phi_s2s, self.phi_ss)

    @property
    def sigma(self) -> float:
        """float: Total standard deviation σ = √(τ² + φ_S2S² + φ_SS²)."""
        return math.hypot(self.tau, self.phi_s2s, self.phi_ss)


def _theta(tau2: float, phi_s2s2: float, phi_ss2: float) -> tuple[float, float]:
    """Relative stds (τ/φ_SS, φ_S2S/φ_SS) of the variances, which must be valid."""
    if not (phi_ss2 > 0 and tau2 >= 0 and phi_s2s2 >= 0) or math.isinf(
        tau2 + phi_s2s2 + phi_ss2
    ):
        raise ValueError("variances must be finite, with τ², φ_S2S² ≥ 0 and φ_SS² > 0")
    return math.sqrt(tau2 / phi_ss2), math.sqrt(phi_s2s2 / phi_ss2)


def _oriented[T](pair: tuple[T, T], flipped: bool) -> tuple[T, T]:
    """Reorder an (event, site) pair to (kept, eliminated), or back."""
    return (pair[1], pair[0]) if flipped else pair


class _Stats(NamedTuple):
    """Statistics of r = a + u_k[k] + u_d[d] + ε, for kept k and eliminated d."""

    k: np.ndarray
    d: np.ndarray
    n_k: np.ndarray  # records per kept level
    n_d: np.ndarray  # records per eliminated level
    n_kd: np.ndarray  # (K, D) record counts
    sum_k: np.ndarray  # residual sum per kept level
    sum_d: np.ndarray  # residual sum per eliminated level
    rr: float  # rᵀr


def _stats(
    r: np.ndarray, e: np.ndarray, s: np.ndarray, n_events: int, n_sites: int
) -> tuple[_Stats, bool]:
    """Statistics with the grouping of fewer levels kept dense, flipped if sites."""
    flipped = n_events > n_sites
    k, d = _oriented((e, s), flipped)
    K, D = _oriented((n_events, n_sites), flipped)
    n_kd = np.bincount(k * D + d, minlength=K * D).reshape(K, D).astype(float)
    sum_k = np.bincount(k, weights=r, minlength=K)
    sum_d = np.bincount(d, weights=r, minlength=D)
    n_k, n_d = n_kd.sum(axis=1), n_kd.sum(axis=0)
    return _Stats(k, d, n_k, n_d, n_kd, sum_k, sum_d, float(r @ r)), flipped


class _Fit(NamedTuple):
    """Solution [a, v_k, v_d] of the mixed-model equations at one θ."""

    neg2_reml: float
    phi_ss2: float
    x: np.ndarray  # [a, v_k]
    v_d: np.ndarray
    w: np.ndarray  # inverse diagonal of the eliminated block
    L: np.ndarray  # lower Cholesky factor of the Schur complement W


def _fit(st: _Stats, t_k: float, t_d: float, phi_ss2: float | None = None) -> _Fit:
    """Solve M [a, v] = [X ZΛ]ᵀr at relative stds (t_k, t_d), with u = Λv.

    M = [X ZΛ]ᵀ[X ZΛ] + diag(0, 1, …, 1). Its eliminated block is the diagonal
    1/w, so only the Schur complement W of size 1 + K needs factorising, and W
    comes straight from the counts. Without `phi_ss2`, φ_SS² takes its
    closed-form REML value.
    """
    n_k, n_d, n_kd = st.n_k, st.n_d, st.n_kd
    w = 1 / (t_d**2 * n_d + 1)
    nw = n_kd * w
    W = np.diag(
        np.concatenate([[n_d.sum() - t_d**2 * (w * n_d) @ n_d], t_k**2 * n_k + 1])
    )
    W[0, 1:] = W[1:, 0] = t_k * (n_k - t_d**2 * nw @ n_d)
    W[1:, 1:] -= (t_k * t_d) ** 2 * nw @ n_kd.T

    c_x = np.concatenate([[st.sum_k.sum()], t_k * st.sum_k])  # [X ZΛ]ᵀr, kept rows
    c_d = t_d * st.sum_d  # [X ZΛ]ᵀr, eliminated rows
    L = np.linalg.cholesky(W)
    reduced = c_x - t_d * np.concatenate([[(w * n_d) @ c_d], t_k * nw @ c_d])
    x = scipy.linalg.cho_solve((L, True), reduced)
    v_d = w * (c_d - t_d * (n_d * x[0] + t_k * n_kd.T @ x[1:]))

    # Penalised residual sum of squares ‖r − Xa − ZΛv‖² + ‖v‖² = rᵀr − [x, v_d]ᵀc.
    # −2ℓ_REML = (N − 1)·log φ_SS² + log|M| + q/φ_SS², minimised in closed form
    # by φ_SS² = q/(N − 1). Equals log|V| + log|XᵀV⁻¹X| + rᵀPr.
    q = st.rr - x @ c_x - v_d @ c_d
    N = n_d.sum()
    if phi_ss2 is None:
        phi_ss2 = q / (N - 1)
    logdet_M = 2 * np.log(np.diag(L)).sum() - np.log(w).sum()
    neg2 = (N - 1) * np.log(phi_ss2) + logdet_M + q / phi_ss2
    return _Fit(float(neg2), float(phi_ss2), x, v_d, w, L)


def _decomposition(
    r: np.ndarray,
    e: np.ndarray,
    s: np.ndarray,
    events: np.ndarray,
    sites: np.ndarray,
    theta: tuple[float, float],
    phi_ss2: float | None = None,
) -> Decomposition:
    """Solve at θ = (τ/φ_SS, φ_S2S/φ_SS), with the conditional stds."""
    # Centre r: the solve forms q = rᵀr − [x, v_d]ᵀc, which cancels badly when
    # the residuals sit far from zero. The mean goes back into the bias.
    centre = r.mean()
    r = r - centre
    st, flipped = _stats(r, e, s, len(events), len(sites))
    t_k, t_d = _oriented(theta, flipped)
    fit = _fit(st, t_k, t_d, phi_ss2)
    u_k, u_d = t_k * fit.x[1:], t_d * fit.v_d

    # Conditional covariance of [a, u] is φ_SS² Λ̃ M⁻¹ Λ̃, Λ̃ = diag(1, Λ). Its
    # diagonal comes from the Schur blocks, with B the off-diagonal block of M.
    W_inv = scipy.linalg.cho_solve((fit.L, True), np.eye(len(fit.x)))
    BD = t_d * np.vstack([st.n_d, t_k * st.n_kd]) * fit.w
    var_vd = fit.w + np.sum((W_inv @ BD) * BD, axis=0)
    std_k = t_k * np.sqrt(fit.phi_ss2 * np.diag(W_inv)[1:])
    std_d = t_d * np.sqrt(fit.phi_ss2 * var_vd)

    event_terms, site_terms = _oriented((u_k, u_d), flipped)
    event_std, site_std = _oriented((std_k, std_d), flipped)
    remaining = r - fit.x[0] - u_k[st.k] - u_d[st.d]
    phi_ss = math.sqrt(fit.phi_ss2)
    return Decomposition(
        bias=float(centre + fit.x[0]),
        bias_std=math.sqrt(fit.phi_ss2 * W_inv[0, 0]),
        tau=float(theta[0] * phi_ss),
        phi_s2s=float(theta[1] * phi_ss),
        phi_ss=phi_ss,
        events=events,
        event_terms=event_terms,
        event_std=event_std,
        sites=sites,
        site_terms=site_terms,
        site_std=site_std,
        within_event=remaining + site_terms[s],
        remaining=remaining,
        neg2_reml=fit.neg2_reml,
    )


def solve_mme(
    residuals: np.ndarray,
    event_idx: np.ndarray,
    site_idx: np.ndarray,
    tau2: float,
    phi_s2s2: float,
    phi_ss2: float,
) -> Decomposition:
    """Solve the mixed-model equations at fixed variances.

    Parameters
    ----------
    residuals : np.ndarray
        Residual per record.
    event_idx, site_idx : np.ndarray
        Integer event and site code per record: 0 to n − 1, with every code
        present.
    tau2, phi_s2s2, phi_ss2 : float
        The variances τ², φ_S2S² and φ_SS².

    Returns
    -------
    Decomposition
        The terms at these variances, with ids 0 to n − 1.

    Raises
    ------
    ValueError
        If a code is missing, negative or not an integer, or a variance is
        invalid.
    """
    theta = _theta(tau2, phi_s2s2, phi_ss2)
    e, s = np.asarray(event_idx), np.asarray(site_idx)
    for idx in (e, s):
        if idx.dtype.kind not in "iu" or idx.min() < 0 or not np.bincount(idx).all():
            raise ValueError("indices must be integer codes 0 to n − 1, each present")
    events, sites = np.arange(e.max() + 1), np.arange(s.max() + 1)
    r = np.asarray(residuals, dtype=float)
    return _decomposition(r, e, s, events, sites, theta, phi_ss2)


def decompose(
    residuals: np.ndarray,
    event_ids: np.ndarray,
    site_ids: np.ndarray,
    initial_variances: tuple[float, float, float] | None = None,
) -> Decomposition:
    """Decompose residuals into bias, δB_e, δS2S_s and δWS_es by REML.

    Parameters
    ----------
    residuals : np.ndarray
        Residual ln(observed) − ln(predicted) per record. Each must be finite,
        so drop missing records first.
    event_ids, site_ids : np.ndarray
        Event and site label per record, of any sortable dtype.
    initial_variances : tuple of float, optional
        Starting (τ², φ_S2S², φ_SS²), such as the fit at a neighbouring
        intensity measure. Defaults to τ = φ_S2S = φ_SS.

    Returns
    -------
    Decomposition
        The bias, standard deviations and terms of the fitted model.

    Raises
    ------
    ValueError
        If the inputs differ in length, a residual isn't finite, an id is
        missing (NaN), the residuals have zero variance, there are fewer than two
        events or two sites, or `initial_variances` is invalid.
    RuntimeError
        If the optimiser fails, or τ/φ_SS or φ_S2S/φ_SS reaches its cap of
        1000. The cap means the remaining residual is close to noise-free, so
        the data can't determine φ_SS.
    """
    r = np.asarray(residuals, dtype=float)
    event_ids, site_ids = np.asarray(event_ids), np.asarray(site_ids)
    if r.ndim != 1 or event_ids.shape != r.shape or site_ids.shape != r.shape:
        raise ValueError(
            "residuals, event_ids and site_ids must be 1-D and equal length"
        )
    if not np.all(np.isfinite(r)):
        raise ValueError("residuals must be finite; drop missing records first")
    scale = r.std()
    if np.ptp(r) == 0 or not scale > 0:  # constant, or a variance that underflows
        raise ValueError("residuals have no variance to decompose")
    events, e = np.unique(event_ids, return_inverse=True)
    sites, s = np.unique(site_ids, return_inverse=True)
    if any(isinstance(x, float) and math.isnan(x) for x in (*events, *sites)):
        raise ValueError("event and site ids must not be missing (NaN)")
    if len(events) < 2 or len(sites) < 2:
        raise ValueError("need at least two events and two sites")

    # θ is scale-free, so optimise on standardised residuals: shifted or
    # rescaled residuals then give the optimiser the same problem up to rounding.
    st, flipped = _stats((r - r.mean()) / scale, e, s, len(events), len(sites))
    theta0 = (1.0, 1.0) if initial_variances is None else _theta(*initial_variances)

    # −2ℓ depends only on θ², so optimise over θ ∈ ℝ² and take |θ|. This makes
    # θ = 0 an interior point. As a bound it stalls optimisers, because the
    # gradient vanishes there. Derivative-free, as lme4.
    def to_theta(x: np.ndarray) -> np.ndarray:
        return np.minimum(np.abs(x), _THETA_MAX)

    # The objective is −2ℓ per record. Rounding in −2ℓ grows with N, so this
    # keeps it below the absolute `fatol` that ends the search.
    result = scipy.optimize.minimize(
        lambda x: _fit(st, *to_theta(x)).neg2_reml / len(r),
        to_theta(np.array(_oriented(theta0, flipped))),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 2000},
    )
    t_k, t_d = to_theta(result.x)
    if max(t_k, t_d) >= _THETA_MAX:
        raise RuntimeError(
            "φ_SS is not identifiable: τ/φ_SS or φ_S2S/φ_SS reached the cap of "
            f"{_THETA_MAX:g}, so the remaining residual is close to noise-free"
        )
    if not result.success:
        raise RuntimeError(f"REML optimisation failed: {result.message}")
    theta = _oriented((float(t_k), float(t_d)), flipped)
    return _decomposition(r, e, s, events, sites, theta)
