# Revati Integration Design for Veeksha

## Overview

This document describes the integration of Revati time synchronization into the Veeksha benchmarking framework. The integration enables deterministic, reproducible benchmarks with virtual time control for LLM serving systems.

## Background

### What is Revati?

Revati is a time emulation/simulation framework that provides:
- **Virtual time control**: Replace wall-clock time with synchronized virtual time
- **Time coordination**: Coordinate time advancement across multiple clients
- **Event loop integration**: Automatic tracking of async I/O operations
- **Deterministic replay**: Reproduce exact timing behavior for debugging

### Why Integrate with Veeksha?

Veeksha is a benchmark framework for LLM serving systems that:
- Generates synthetic request workloads with configurable patterns
- Measures throughput, latency, and other performance metrics
- Uses multiple worker processes with asyncio for concurrent requests

**Benefits of integration:**
1. **Reproducibility**: Run the same benchmark multiple times with identical timing
2. **Time compression**: Speed up benchmarks by skipping idle periods
3. **Debugging**: Replay specific timing scenarios that caused issues
4. **Analysis**: Separate computation time from wall-clock time

## Architecture Comparison

### Veeksha vs SGLang Architecture

The integration differs from the SGLang bench_serving integration due to architectural differences:

| Aspect | SGLang bench_serving | Veeksha |
|--------|---------------------|---------|
| **Dispatch model** | Single asyncio event loop | Thread-based dispatch |
| **Request execution** | Same event loop | Multi-process workers with separate asyncio loops |
| **Timing control** | `await asyncio.sleep()` | `time.sleep()` |
| **Revati client** | `AsyncClient` | `Client` (synchronous) |

### Veeksha Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Main Process                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  run_main_loop()                                       │ │
│  │  ┌──────────────────┐      ┌──────────────────────┐   │ │
│  │  │ Dispatch Thread  │      │ Process Thread       │   │ │
│  │  │                  │      │                      │   │ │
│  │  │ - Generate reqs  │      │ - Collect metrics    │   │ │
│  │  │ - Schedule timing│      │ - Process responses  │   │ │
│  │  │ - Revati client  │      │                      │   │ │
│  │  │   time_jump()    │      │                      │   │ │
│  │  └──────────────────┘      └──────────────────────┘   │ │
│  └─────────┬───────────────────────────┬──────────────────┘ │
│            │ input_queue               │ output_queue       │
└────────────┼───────────────────────────┼────────────────────┘
             │                           │
             ▼                           │
┌─────────────────────────────────────────────────────────────┐
│              Worker Processes (separate OS processes)       │
│  ┌────────────────┐  ┌────────────────┐  ┌───────────────┐ │
│  │ Worker 1       │  │ Worker 2       │  │ Worker N      │ │
│  │ ┌────────────┐ │  │ ┌────────────┐ │  │ ┌───────────┐ │ │
│  │ │ asyncio    │ │  │ │ asyncio    │ │  │ │ asyncio   │ │ │
│  │ │ event loop │ │  │ │ event loop │ │  │ │ event loop│ │ │
│  │ │            │ │  │ │            │ │  │ │           │ │ │
│  │ │ - aiohttp  │ │  │ │ - aiohttp  │ │  │ │ - aiohttp │ │ │
│  │ │ - HTTP reqs│ │  │ │ - HTTP reqs│ │  │ │ - HTTP reqs│││
│  │ └────────────┘ │  │ └────────────┘ │  │ └───────────┘ │ │
│  └────────────────┘  └────────────────┘  └───────────────┘ │
│         │                    │                    │         │
└─────────┼────────────────────┼────────────────────┼─────────┘
          │                    │                    │
          ▼                    ▼                    ▼
     LLM Server          LLM Server           LLM Server
```

## Integration Design

### Key Components

#### 1. Dispatch Thread Time Control

**Location**: `veeksha/benchmark.py:dispatch_requests()`

**Functionality**: The dispatch thread controls request timing using either:
- `time.sleep(sleep_time)` - Normal mode
- `revati_client.time_jump(sleep_time)` - Revati mode

**Code**:
```python
def dispatch_requests(
    ...,
    revati_client: Optional["RevatiClient"] = None,
) -> None:
    while not stop_event.is_set():
        # ... request scheduling logic ...

        time_until = scheduler.time_until_next_ready()
        sleep_time = 0.01 if time_until is None else min(max(time_until, 0.0), 0.1)

        # Use revati time synchronization if available
        if revati_client is not None:
            revati_client.time_jump(sleep_time)
        else:
            time.sleep(sleep_time)
```

#### 2. Client Lifecycle Management

**Location**: `veeksha/benchmark.py:run_main_loop()`

**Lifecycle stages**:
1. **Initialization**: Create client, connect, register
2. **Simulation start**: Begin coordinated timing
3. **Dispatch operation**: Control timing via time_jump()
4. **Finalization**: Remove from coordination
5. **Cleanup**: Disconnect

**Code**:
```python
def run_main_loop(...):
    # Initialize revati client if enabled
    revati_client = None
    if enable_revati and RevatiClient:
        revati_client = RevatiClient(
            server_address=revati_server_address,
            client_name="Dispatch"
        )
        revati_client.connect()
        revati_client.register()
        revati_client.start_simulation()

    # ... run benchmark ...

    # Clean up revati client
    if revati_client is not None:
        revati_client.finalize()
        revati_client.disconnect()
```

#### 3. Worker Process Time Control

**Location**: Automatic via `revati-run`

**Functionality**: Worker processes run separate asyncio event loops that make HTTP requests. When launched with `revati-run --enable-event-loop-integration`:
- Each asyncio loop is automatically tracked as a separate client
- I/O operations (aiohttp, asyncio.sleep) are intercepted
- Time advances when all workers are blocked or complete

**No code changes needed** - handled by revati's LD_PRELOAD interception.

### Configuration

#### CLI Arguments

Added to `veeksha/config/benchmark.py:BenchmarkConfig`:

```python
enable_revati_client: bool = field(
    default=False,
    metadata={
        "help": "Enable Revati client for time synchronization. "
        "Requires revati-server and revati-run wrapper."
    },
)

revati_server_address: str = field(
    default="tcp://localhost:5555",
    metadata={
        "help": "Address of Revati server. Used when enable_revati_client is True."
    },
)
```

#### Usage

```bash
# Start revati server
revati-server &

# Run benchmark with revati integration
revati-run --enable-event-loop-integration python -m veeksha.benchmark \
  --enable_revati_client \
  --revati_server_address tcp://localhost:5555 \
  --api_url http://localhost:30000/generate \
  --tokenizer meta-llama/Llama-3.1-8B-Instruct \
  --num_prompts 100 \
  --request_rate 10.0
```

## Design Decisions

### 1. Why Synchronous Client in Dispatch Thread?

**Decision**: Use `Client` (synchronous) instead of `AsyncClient`

**Rationale**:
- Dispatch thread uses `time.sleep()`, not `await asyncio.sleep()`
- Thread-based, not async-based
- Can't use `await` in non-async context
- `Client.time_jump()` is synchronous and thread-safe

**Alternative considered**: Convert dispatch to async
- **Rejected**: Would require major refactoring of threading model
- **Complexity**: Would need to coordinate between async and multiprocessing

### 2. Why No Revati Client in Worker Processes?

**Decision**: Workers automatically tracked via event loop integration

**Rationale**:
- Each worker runs in separate OS process
- Workers use asyncio extensively for HTTP requests
- Manual client management would be complex and error-prone
- Event loop integration handles this automatically

**How it works**:
1. `revati-run --enable-event-loop-integration` intercepts epoll/select calls
2. Each asyncio loop creates its own client automatically
3. Clients are named `asyncio_loop_<pid>_<epoll_fd>`
4. Time advances when all loops are idle or complete

### 3. Client Finalization

**Decision**: Always finalize dispatch client before cleanup

**Rationale**:
- Dispatch client controls timing but doesn't do heavy I/O
- After dispatching all requests, it's idle
- Must finalize to avoid blocking worker time advancement
- Similar to SGLang integration pattern

### 4. Error Handling

**Decision**: Graceful fallback to normal timing

**Rationale**:
- If Revati client init fails, warn and continue with time.sleep()
- Allows benchmarks to run without Revati if needed
- Makes integration non-breaking for existing users

## Time Semantics

### Dispatch Client Time Jumps

The dispatch client uses **relative time jumps**:

```python
revati_client.time_jump(sleep_time)  # Jump forward by sleep_time seconds
```

**Behavior**:
- Blocks until server advances time by `sleep_time` (or more)
- Server coordinates with worker event loops
- Returns when new time reached

### Worker Event Loop Integration

Workers use **I/O-based time advancement**:

- `aiohttp` HTTP request: Blocks until response received in virtual time
- `asyncio.sleep(duration)`: Blocks for `duration` virtual seconds
- No explicit time_jump() calls needed

**Time advancement algorithm**:
1. All clients (dispatch + workers) submit time jump requests
2. Server advances to minimum requested time
3. Clients with that time are woken up
4. Repeat until simulation complete

## Integration Points

### Files Modified

1. **veeksha/benchmark.py**
   - Import `RevatiClient`
   - Add `revati_client` parameter to `dispatch_requests()`
   - Replace `time.sleep()` with conditional `time_jump()`
   - Add client lifecycle management in `run_main_loop()`

2. **veeksha/config/benchmark.py**
   - Add `enable_revati_client` field
   - Add `revati_server_address` field

### Dependencies

**Runtime requirements**:
- `revati` Python package installed
- `revati-server` process running
- Benchmark launched via `revati-run` wrapper with `--enable-event-loop-integration`

**Build requirements**:
- No additional build dependencies
- Imports are optional (graceful fallback if not available)

## Comparison with SGLang Integration

| Aspect | SGLang | Veeksha |
|--------|--------|---------|
| **Architecture** | Single async event loop | Thread + multi-process |
| **Client type** | `AsyncClient` | `Client` (sync) |
| **Client location** | Main async function | Dispatch thread |
| **Dispatch timing** | `await client.time_jump()` | `client.time_jump()` (sync) |
| **Worker tracking** | Manual client per worker | Automatic via event loop integration |
| **Integration complexity** | Low (single event loop) | Medium (thread + process coordination) |

## Limitations and Future Work

### Current Limitations

1. **No wall time measurement**: Virtual time ≠ wall time
   - Need separate wall time tracking for speedup metrics

2. **Process synchronization**: Workers in separate processes
   - Can't share Python objects directly
   - Communication via queues only

3. **Request rate semantics**: Request rate is virtual, not wall-clock
   - 10 req/s means 10 requests per virtual second
   - Actual wall-clock rate may be much higher

### Future Enhancements

1. **Wall time comparison**: Add wall time tracking to measure speedup ratio
   ```python
   wall_start = time.perf_counter()
   virtual_start = revati_client.get_virtual_time()
   # ... benchmark ...
   speedup = (wall_end - wall_start) / (virtual_end - virtual_start)
   ```

2. **Chrome trace integration**: Generate timeline visualization
   - Already supported by revati via `REVATI_CHROME_TRACE` env var
   - Need to document usage pattern

3. **Metrics annotation**: Tag metrics with virtual vs wall time
   - Separate throughput metrics
   - Add "virtual time" and "wall time" columns to output

## References

- Revati documentation: `docs/design/time_sync/`
- SGLang integration: `sglang/python/sglang/bench_serving.py`
- Veeksha architecture: `veeksha/README.md`
