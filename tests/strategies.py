import numpy as np
from hypothesis import strategies as st

residual_values = st.floats(-5, 5, allow_subnormal=False)
variances = st.tuples(*[st.floats(1e-3, 4.0)] * 3)


@st.composite
def designs(draw, min_extra=0, max_extra=20):
    """(residuals, event_idx, site_idx) with every event and site recorded at least once."""
    E = draw(st.integers(2, 6))
    S = draw(st.integers(2, 6))
    pairs = (
        [(i, draw(st.integers(0, S - 1))) for i in range(E)]
        + [(draw(st.integers(0, E - 1)), j) for j in range(S)]
        + draw(
            st.lists(
                st.tuples(st.integers(0, E - 1), st.integers(0, S - 1)),
                min_size=min_extra,
                max_size=max_extra,
            )
        )
    )
    e, s = (np.array(x) for x in zip(*pairs))
    r = np.array(draw(st.lists(residual_values, min_size=len(e), max_size=len(e))))
    return r, e, s


# Replicated designs (more records than unknowns) with non-constant residuals,
# so the REML optimum exists and the invariance properties apply.
identifiable_designs = designs(min_extra=15, max_extra=35).filter(
    lambda d: d[0].std() > 0
)
