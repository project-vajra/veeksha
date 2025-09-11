DEFAULT_SEED = 42
ZIPF_REQUEST_GENERATOR_EPS = 1e-8
ALLOWED_TS_UNITS = {"ms", "s"}
SCALE_TO_SECONDS = {
    "ms": 1e-3,
    "s": 1.0,
}
ALLOWED_EXHAUSTION_POLICIES = {"error", "stop", "wrap"}
