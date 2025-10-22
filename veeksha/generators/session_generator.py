import copy
import os
from typing import Dict, List, cast

import numpy as np
import pandas as pd

from veeksha.config.generators.interval_generator.gamma_generator import (
    GammaRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.interval_generator.poisson_generator import (
    PoissonRequestIntervalGeneratorConfig,
)
from veeksha.config.generators.session_generator import (
    SessionGeneratorConfig,
)
from veeksha.core.seeding import SeedManager
from veeksha.generators.interval_generator.generator_registry import (
    RequestIntervalGeneratorRegistry,
)
from veeksha.logger import init_logger

logger = init_logger(__name__)


class TrieNode:
    def __init__(self):
        self.children: Dict[int, TrieNode] = {}


class PrefixCache:
    def __init__(self):
        self.root = TrieNode()

    def add_request(self, request):
        # insert the full request into the trie
        current = self.root
        for id in request["hash_ids"]:
            if id not in current.children:
                current.children[id] = TrieNode()
            current = current.children[id]

    def get_prefix_length_match(self, request):
        # traverse trie to find longest cached prefix
        current = self.root
        length = 0
        for id in request["hash_ids"]:
            if id in current.children:
                current = current.children[id]
                length += 1
            else:
                break
        return length


class SessionGenerator:
    def __init__(
        self,
        config: SessionGeneratorConfig,
        seed_manager: SeedManager,
    ):
        self.config = config
        self.seed_manager = seed_manager
        self.rng_factory = seed_manager.numpy_factory("sessions")
        self.session_interval_generator = RequestIntervalGeneratorRegistry.get(
            self.config.session_interval_generator_config.get_type(),
            self.config.session_interval_generator_config,
            rng=self.rng_factory(),
        )

    @staticmethod
    def create_running_hashes(hash_ids: List[int]):
        """Create running hashes for a sequence of hash IDs.
        Each running hash at position i is a hash of the sequence from 0 to i."""
        if not hash_ids:
            return []

        running_hashes = []

        # For each position i, create a hash of the tuple of all elements from 0 to i
        for i in range(len(hash_ids)):
            # Create a tuple of the prefix and hash it
            prefix_tuple = tuple(hash_ids[: i + 1])
            prefix_hash = hash(prefix_tuple) % (2**32)  # Keep hash size manageable
            running_hashes.append(prefix_hash)

        return running_hashes

    def rejection_sample(self, remaining_sessions, current_timestamp, rng=None):
        """Rejection sample a session.

        Args:
            remaining_sessions: List of remaining sessions to sample from
            current_timestamp: Current timestamp in seconds
        """
        assert remaining_sessions, "No sessions remaining to sample from"

        rng = rng or self.rng_factory()

        next_interval = self.session_interval_generator.get_next_inter_request_time()

        # Rejection sampling to bias towards sessions with more requests
        max_iterations = 1000  # prevent infinite loops
        iteration_count = 0
        session = None

        while iteration_count < max_iterations:
            # Propose a session randomly from remaining sessions
            proposed_idx = rng.randint(0, len(remaining_sessions))
            proposed_session = remaining_sessions[proposed_idx]

            acceptance_prob = len(proposed_session) / self.config.max_session_size
            if rng.random() < acceptance_prob:
                session = remaining_sessions.pop(proposed_idx)
                break

            iteration_count += 1

        # fallback: take a random session if max iterations reached
        if session is None:
            proposed_idx = rng.randint(0, len(remaining_sessions))
            session = remaining_sessions.pop(proposed_idx)

        session_original_timestamp = None
        for request in session:
            if session_original_timestamp is None:
                session_original_timestamp = request["timestamp"]
            request["timestamp"] = current_timestamp + (
                request["timestamp"] - session_original_timestamp
            )

        current_timestamp += next_interval

        return session, current_timestamp

    def save_requests_as_trace(self, requests_df: pd.DataFrame, save_suffix: str = ""):
        """Save the trace to a jsonl trace file.

        Args:
            requests_df: DataFrame with timestamps in milliseconds (trace file format)
            save_suffix: Optional suffix to append to the filename (before extension)
        """

        # append config params to file name
        def create_clean_filename():
            base_name = (
                self.config.trace_file_name
                if self.config.trace_file_name
                else "session_trace"
            )

            params = [
                f"prefix-{self.config.minimum_prefix_match}",
                f"min-{self.config.min_session_size}",
                f"max-{self.config.max_session_size}",
                f"interval-{self.config.max_request_interval}",
            ]

            interval_config = self.config.session_interval_generator_config

            if isinstance(
                interval_config,
                (
                    PoissonRequestIntervalGeneratorConfig,
                    GammaRequestIntervalGeneratorConfig,
                ),
            ):
                params.append(f"qps-{interval_config.qps}")

            suffix = save_suffix if save_suffix else ""
            return f"{base_name}_{'_'.join(params)}{suffix}.jsonl"

        target_dir = self.config.trace_file_save_dir
        file_name = os.path.join(target_dir, create_clean_filename())
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        requests_df.to_json(file_name, orient="records", lines=True)
        logger.info(f"Saved generated trace with sessions to {file_name}")

    def find_sessions(self, requests):
        """Group requests into sessions based on prefix matching."""
        request_to_session = {}
        hash_to_request = {}
        hash_to_length = {}
        sessions = {}

        # insert all the hash_ids into the hash_to_request
        for idx, request in enumerate(requests):
            hash_ids = request["hash_ids"]
            for hash_id in hash_ids:
                if hash_id not in hash_to_request:
                    hash_to_request[hash_id] = idx

        # Process each request in order
        for idx, request in enumerate(requests):
            current_hashes = request["hash_ids"]
            best_match_hash = None
            best_match_length = 0

            # Check for matches of increasing length
            for prefix_len, hash_id in enumerate(current_hashes):
                # If this prefix exists in our map, we have a match
                if hash_id in hash_to_length:
                    best_match_hash = hash_id
                    best_match_length = prefix_len + 1

            if best_match_length > self.config.minimum_prefix_match * len(
                current_hashes
            ):
                # match to the existing session
                matched_request_idx = hash_to_request[best_match_hash]
                session_id = request_to_session[matched_request_idx]
                request_to_session[idx] = session_id
                request["session_id"] = session_id
                sessions[session_id].append(request)
            else:
                # create a new session
                request_to_session[idx] = idx
                request["session_id"] = idx
                sessions[idx] = [request]

            # Add all prefixes of the current hash sequence to our map
            for prefix_len, hash_id in enumerate(current_hashes):
                hash_to_length[hash_id] = prefix_len + 1

        # Enforce maximum time between requests in a session
        MAX_SESSION_GAP = self.config.max_request_interval  # Keep in seconds

        new_sessions = {}
        next_session_id = 0

        # Process each original session
        for session_id, session in sessions.items():
            # Sort requests within each session by timestamp
            sorted_session = sorted(session, key=lambda x: x["timestamp"])

            current_session = []
            prev_timestamp = None
            current_new_session_id = next_session_id

            for request in sorted_session:
                # If this is the first request in the session or the gap is acceptable
                if (
                    prev_timestamp is None
                    or (request["timestamp"] - prev_timestamp) <= MAX_SESSION_GAP
                ):
                    # Add to current session
                    current_session.append(request)
                else:
                    # Gap is too large, finish current session and start a new one
                    if current_session:
                        new_sessions[current_new_session_id] = current_session
                        next_session_id += 1
                        current_new_session_id = next_session_id
                        current_session = [
                            request
                        ]  # Start new session with current request

                # Update request's session_id to the new one
                request["session_id"] = current_new_session_id
                prev_timestamp = request["timestamp"]

            # Add the last session if it has requests
            if current_session:
                new_sessions[current_new_session_id] = current_session
                next_session_id += 1

        return new_sessions

    def sample_sessions(self, sessions):
        """Sample sessions using dispatch rate with poisson distribution."""
        sessions_list = list(sessions.values())

        # List to track which sessions have been sampled
        remaining_sessions = sessions_list.copy()

        timestamp = 0  # Start at time 0 (in seconds)
        sampled_sessions = []

        session_id = 0
        current_rng = self.rng_factory()
        while remaining_sessions:
            session, timestamp = self.rejection_sample(
                remaining_sessions, timestamp, current_rng
            )
            # Make a deep copy of the session to avoid modifying the original data
            session_copy = copy.deepcopy(session)
            # Assign session_id to each request in the session
            for request in session_copy:
                request["session_id"] = session_id
            sampled_sessions.append(session_copy)
            session_id += 1

        return sampled_sessions

    def generate_sessions(self, requests_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate sessions, creating a column `session_id` in the input DataFrame.

        Args:
            requests_df: DataFrame containing the requests with timestamps in seconds

        Returns:
            DataFrame with sessions and timestamps in seconds
        """
        # Store the original hash IDs before replacing them
        requests_df["original_hash_ids"] = requests_df["hash_ids"]

        # Replace hash_ids with running hashes
        requests_df["hash_ids"] = requests_df.apply(
            lambda row: self.create_running_hashes(row["original_hash_ids"]), axis=1
        )

        # Convert the df into list of dict
        requests = requests_df.to_dict("records")

        # Get sessions based on prefix matches
        sessions = self.find_sessions(requests)
        logger.info(f"Created {len(sessions)} sessions for {len(requests)} requests")

        # Add metadata: session id, number of requests in session
        for session_id, session in sessions.items():
            for request in session:
                request["session_id"] = session_id
                request["num_requests_in_session"] = len(session)

        # Filter sessions and requests
        sessions_to_delete = []
        sessions_to_delete_lt_min = 0

        for session_id, session in sessions.items():
            # If there are sessions with < min_session_size requests, delete them
            if len(session) < self.config.min_session_size:
                sessions_to_delete.append(session_id)
                sessions_to_delete_lt_min += 1

        logger.info(
            f"Deleted {sessions_to_delete_lt_min} sessions with less than {self.config.min_session_size} requests"
        )

        # Delete sessions marked for deletion
        for session_id in sessions_to_delete:
            del sessions[session_id]

        # Delete sessions with more than max_session_size requests
        sessions_to_delete = [
            session_id
            for session_id, session in sessions.items()
            if len(session) > self.config.max_session_size
        ]
        for session_id in sessions_to_delete:
            del sessions[session_id]

        if sessions_to_delete:
            logger.info(
                f"Deleted {len(sessions_to_delete)} sessions with more than {self.config.max_session_size} requests"
            )

        # logger.info session stats before sampling
        session_lengths = [len(session) for session in sessions.values()]
        if session_lengths:
            logger.info("=====SESSION STATS BEFORE SESSION SAMPLING=====")
            logger.info(f"Number of sessions: {len(sessions)}")
            logger.info(f"Number of requests: {sum(session_lengths)}")
            logger.info(f"Session max length: {max(session_lengths)}")
            logger.info(f"Session min length: {min(session_lengths)}")
            logger.info(f"Session mean length: {np.mean(session_lengths):.2f}")
            logger.info(f"Session std length: {np.std(session_lengths):.2f}")
            logger.info(f"Session median length: {np.median(session_lengths):.2f}")
            logger.info(f"Session p25 length: {np.percentile(session_lengths, 25):.2f}")
            logger.info(f"Session p75 length: {np.percentile(session_lengths, 75):.2f}")
            logger.info(f"Session p90 length: {np.percentile(session_lengths, 90):.2f}")
            logger.info("-" * 30)
        else:
            logger.info("No valid sessions found after filtering")
            return requests_df

        # Sample sessions using dispatch rate
        sampled_sessions = self.sample_sessions(sessions)

        # logger.info session stats after sampling
        session_lengths = [len(session) for session in sampled_sessions]
        logger.info("=====SESSION STATS AFTER SESSION SAMPLING=====")
        logger.info(f"Number of sessions: {len(sampled_sessions)}")
        logger.info(f"Number of requests: {sum(session_lengths)}")
        logger.info(f"Session max length: {max(session_lengths)}")
        logger.info(f"Session min length: {min(session_lengths)}")
        logger.info(f"Session mean length: {np.mean(session_lengths):.2f}")
        logger.info(f"Session std length: {np.std(session_lengths):.2f}")
        logger.info(f"Session median length: {np.median(session_lengths):.2f}")
        logger.info(f"Session p25 length: {np.percentile(session_lengths, 25):.2f}")
        logger.info(f"Session p75 length: {np.percentile(session_lengths, 75):.2f}")
        logger.info(f"Session p90 length: {np.percentile(session_lengths, 90):.2f}")
        logger.info("-" * 30)

        # Flatten the sessions
        sampled_requests = [
            request for session in sampled_sessions for request in session
        ]

        # Sort by timestamp
        sampled_requests.sort(key=lambda x: x["timestamp"])

        # Add metadata: request id (sequential) and prefix match percentage
        sequential_request_id = 0
        prefix_cache = PrefixCache()
        total_hashes_matched = 0
        total_hashes_seen = 0
        cummulative_prefix_match_pct = 0.0

        for request in sampled_requests:
            request["request_id"] = sequential_request_id
            sequential_request_id += 1

            # Add prefix match
            request["prefix_match_n_hashes"] = prefix_cache.get_prefix_length_match(
                request
            )
            request["prefix_match_pct"] = (
                round(
                    100 * (request["prefix_match_n_hashes"] / len(request["hash_ids"])),
                    2,
                )
                if request["hash_ids"]
                else 0
            )
            prefix_cache.add_request(request)

            total_hashes_matched += request["prefix_match_n_hashes"]
            total_hashes_seen += len(request["hash_ids"])

            cummulative_prefix_match_pct = (
                round(100 * (total_hashes_matched / total_hashes_seen), 2)
                if total_hashes_seen > 0
                else 0
            )
            request["cummulative_prefix_match_pct"] = cummulative_prefix_match_pct

        logger.info(f"Prefix match in generated trace: {cummulative_prefix_match_pct}%")

        # session dispatch rate in generated trace
        df_tmp = pd.DataFrame(sampled_requests)
        if (
            not df_tmp.empty
            and "session_id" in df_tmp.columns
            and "timestamp" in df_tmp.columns
        ):
            df_tmp = df_tmp.sort_values(by="timestamp")
            first_requests_df = df_tmp.groupby("session_id", as_index=False).first()
            if len(first_requests_df) >= 2:
                duration_s = float(
                    first_requests_df["timestamp"].max()
                    - first_requests_df["timestamp"].min()
                )
                session_dispatch_rate = (
                    (len(first_requests_df) / duration_s) if duration_s > 0 else 0.0
                )
            else:
                session_dispatch_rate = 0.0
            logger.info(
                f"Session dispatch rate in generated trace: {session_dispatch_rate:.6f} sessions/s"
            )
            # Localized diagnostics to catch non-uniform dispatch behavior
            try:
                first_times = np.sort(
                    first_requests_df["timestamp"].to_numpy(dtype=float)
                )
                n_first = len(first_times)
                if n_first >= 2:
                    inter_arrivals = np.diff(first_times)
                    logger.info(
                        "Session first-request inter-arrival (s): "
                        f"count={len(inter_arrivals)}, "
                        f"mean={float(np.mean(inter_arrivals)):.6f}, "
                        f"median={float(np.median(inter_arrivals)):.6f}, "
                        f"p10={float(np.percentile(inter_arrivals, 10)):.6f}, "
                        f"p90={float(np.percentile(inter_arrivals, 90)):.6f}, "
                        f"min={float(np.min(inter_arrivals)):.6f}, "
                        f"max={float(np.max(inter_arrivals)):.6f}"
                    )

                    # Rate over equal-sized time chunks (deciles)
                    total_span = float(first_times[-1] - first_times[0])
                    if total_span > 0:
                        num_chunks = 10
                        edges = np.linspace(
                            first_times[0], first_times[-1], num_chunks + 1
                        )
                        counts, _ = np.histogram(first_times, bins=edges)
                        widths = edges[1:] - edges[:-1]
                        # Avoid division by zero in degenerate bins
                        valid = widths > 0
                        chunk_rates = np.zeros_like(widths, dtype=float)
                        chunk_rates[valid] = counts[valid] / widths[valid]
                        logger.info(
                            "Session dispatch rate by time chunks (/s): "
                            f"min={float(np.min(chunk_rates)):.6f}, "
                            f"p10={float(np.percentile(chunk_rates, 10)):.6f}, "
                            f"median={float(np.median(chunk_rates)):.6f}, "
                            f"p90={float(np.percentile(chunk_rates, 90)):.6f}, "
                            f"max={float(np.max(chunk_rates)):.6f}"
                        )

                        # First half vs second half rate comparison
                        mid_ts = 0.5 * (first_times[0] + first_times[-1])
                        first_half_mask = first_times <= mid_ts
                        n_first_half = int(np.sum(first_half_mask))
                        n_second_half = n_first - n_first_half
                        span_first = float(max(mid_ts - first_times[0], 0.0))
                        span_second = float(max(first_times[-1] - mid_ts, 0.0))
                        rate_first = (
                            (n_first_half / span_first) if span_first > 0 else 0.0
                        )
                        rate_second = (
                            (n_second_half / span_second) if span_second > 0 else 0.0
                        )
                        logger.info(
                            "Session dispatch rate halves (/s): "
                            f"first_half={rate_first:.6f}, second_half={rate_second:.6f}"
                        )

                        # Largest gaps to highlight extreme sparsity
                        if inter_arrivals.size > 0:
                            k = min(3, inter_arrivals.size)
                            largest_gaps = np.sort(inter_arrivals)[-k:][::-1]
                            logger.info(
                                "Largest session start gaps (s): "
                                + ", ".join(f"{g:.6f}" for g in largest_gaps)
                            )
            except Exception:
                # Best-effort diagnostics; do not fail generation due to logging
                pass
        else:
            logger.info("Session dispatch rate in generated trace: N/A")

        # Convert back to DataFrame
        result_df = pd.DataFrame(sampled_requests)

        # Ensure we have all the required columns from the original DataFrame
        for col in requests_df.columns:
            if col not in result_df.columns:
                raise ValueError(f"Column {col} not found in generated trace")

        # Add per-session sequence, within-session gaps, and anchor timestamp for first request
        if (
            not result_df.empty
            and "session_id" in result_df.columns
            and "timestamp" in result_df.columns
        ):

            def _annotate_group(g):
                g = g.sort_values("timestamp").copy()
                g["session_sequence_index"] = range(len(g))
                g["wait_after_prev_response_s"] = g["timestamp"].diff().fillna(0.0)
                g["anchor_at_s"] = None
                if not g.empty:
                    g.loc[g.index[0], "anchor_at_s"] = float(g.iloc[0]["timestamp"])  # type: ignore
                return g

            result_df = cast(
                pd.DataFrame,
                result_df.groupby("session_id", group_keys=False)[
                    list(result_df.columns)
                ].apply(_annotate_group),
            )

        self.trace_df = result_df

        return result_df
