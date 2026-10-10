import os

# Tests use tiny matrices: multithreaded BLAS only adds contention.
for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

from hypothesis import settings

settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=1000, deadline=None, derandomize=True)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "dev"))
