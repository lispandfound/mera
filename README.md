# mera

Mixed-effects decomposition of ground-motion residuals, fitted by restricted
maximum likelihood:

    r_es = a + δB_e + δS2S_s + δWS_es

| Symbol  | Meaning                     | Std       |
|---------|-----------------------------|-----------|
| a       | bias, the fixed intercept   |           |
| δB_e    | between-event term          | τ         |
| δS2S_s  | site-to-site term           | φ_S2S     |
| δWS_es  | remaining residual          | φ_SS      |
| δW_es   | within-event = δS2S + δWS   | φ         |

Events and sites form crossed random effects. The model matches lme4's
`r ~ 1 + (1|event) + (1|site)` with `REML=TRUE`.

```python
from mera import decompose

fit = decompose(residuals, event_ids, site_ids)  # one IM; drop NaNs first
fit.bias, fit.tau, fit.phi_s2s, fit.phi_ss, fit.phi, fit.sigma
fit.events, fit.event_terms, fit.event_std  # δB_e ± conditional std
fit.sites, fit.site_terms, fit.site_std  # δS2S_s ± conditional std
fit.within_event, fit.remaining  # δW_es, δWS_es per record
```

`decompose` raises `RuntimeError` instead of returning a fit it can't trust:
when the optimiser fails, or when τ/φ_SS or φ_S2S/φ_SS reaches its cap of
1000. The cap means the remaining residual is close to noise-free, so the data
can't determine φ_SS.

When looping over IMs, pass the previous fit's variances as
`initial_variances=(tau**2, phi_s2s**2, phi_ss**2)` to warm-start.

The solver eliminates the larger grouping analytically. Its cost scales with
the smaller of the event and site counts. A fit of 1.5M records over
10,000 sites takes seconds.

## Tests

```sh
uv run pytest                          # property tests (Hypothesis) + recovery
HYPOTHESIS_PROFILE=ci uv run pytest    # 1000 examples per property
uv run pytest -m slow                  # large-scale test (~0.3 GB)
```
