"""Constants for worker threads."""

# Queue and polling timeouts
QUEUE_GET_TIMEOUT_S = 0.1
RESULT_POLL_TIMEOUT_S = 0.1
DRAIN_MAX_EMPTY_POLLS = 50  # ~5s with 0.1s timeout
