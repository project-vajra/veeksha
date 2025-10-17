# Revati Integration Testing Guide for Veeksha

## Overview

This document provides comprehensive testing procedures for the Revati time synchronization integration with Veeksha. It covers functional testing, compatibility verification, and debugging strategies.

## Prerequisites

### Software Requirements

1. **Revati installation**:
   ```bash
   # Ensure revati is installed in the same environment
   pip install -e /path/to/revati
   ```

2. **Veeksha dependencies**:
   ```bash
   cd /path/to/veeksha
   pip install -e .
   ```

3. **LLM server** (for end-to-end tests):
   - SGLang, vLLM, or any OpenAI-compatible server
   - Running on localhost:30000 (or adjust URLs accordingly)

### Environment Setup

```bash
# Set up environment
export PYTHONPATH=/path/to/veeksha:/path/to/revati:$PYTHONPATH

# Optional: Enable debug logging
export REVATI_LOGGING_LEVEL=DEBUG
```

## Test Suite

### Test 1: Basic Compatibility Check

**Objective**: Verify Veeksha can import and detect Revati

**Command**:
```bash
python3 -c "
from veeksha.benchmark import RevatiClient
print(f'RevatiClient available: {RevatiClient is not None}')
if RevatiClient:
    print('Revati integration: ✓ AVAILABLE')
else:
    print('Revati integration: ✗ NOT AVAILABLE')
"
```

**Expected output**:
```
RevatiClient available: True
Revati integration: ✓ AVAILABLE
```

**Troubleshooting**:
- If `RevatiClient is None`: Install revati package
- If import fails: Check PYTHONPATH includes revati

---

### Test 2: Configuration Validation

**Objective**: Verify CLI arguments are recognized

**Command**:
```bash
python -m veeksha.benchmark --help | grep -A 2 "revati"
```

**Expected output**:
```
  --enable_revati_client ENABLE_REVATI_CLIENT
                        Enable Revati client for time synchronization...
  --revati_server_address REVATI_SERVER_ADDRESS
                        Address of the Revati timing simulation server...
```

**Troubleshooting**:
- If not found: Check BenchmarkConfig modifications
- Verify frozen_dataclass CLI generation

---

### Test 3: Server Connection Test

**Objective**: Verify dispatch client can connect to revati-server

**Setup**:
```bash
# Terminal 1: Start revati server
pkill -9 -f revati-server  # Clean up any old instances
sleep 2
revati-server -v 2>&1 | tee /tmp/revati_server.log &
sleep 2
```

**Test script** (`test_revati_connection.py`):
```python
from revati.core.time_sync import Client as RevatiClient
import logging

logging.basicConfig(level=logging.DEBUG)

# Test connection
client = RevatiClient(
    server_address="tcp://localhost:5555",
    client_name="TestDispatch"
)

print("Connecting to server...")
client.connect()
print("✓ Connected")

print("Registering client...")
assert client.register()
print("✓ Registered")

print("Starting simulation...")
client.start_simulation()
print("✓ Simulation started")

print("Testing time_jump...")
new_time = client.time_jump(1.0)
print(f"✓ Time advanced to {new_time}")

print("Finalizing...")
client.finalize()
print("✓ Finalized")

print("Disconnecting...")
client.disconnect()
print("✓ Disconnected")

print("\nAll connection tests passed!")
```

**Run**:
```bash
python test_revati_connection.py
```

**Expected output**:
```
Connecting to server...
✓ Connected
Registering client...
✓ Registered
Starting simulation...
✓ Simulation started
Testing time_jump...
✓ Time advanced to 1.0
Finalizing...
✓ Finalized
Disconnecting...
✓ Disconnected

All connection tests passed!
```

**Troubleshooting**:
- **Connection refused**: Server not running or wrong address
- **Registration fails**: Server already has client with same name (restart server)
- **Time jump hangs**: No other clients to coordinate with (expected for single client)

---

### Test 4: Minimal Veeksha Benchmark (No Revati)

**Objective**: Baseline test without Revati integration

**Command**:
```bash
# Create minimal config YAML
cat > test_config_baseline.yaml <<EOF
seed: 42
max_completed_requests: 5
api_url: "http://localhost:30000/v1"
api_key: "test-key"
client_config:
  model: "meta-llama/Llama-3.1-8B-Instruct"
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"
  llm_api: "openai_completions"
  num_clients: 1
  num_concurrent_requests_per_client: 2
request_generator_config:
  request_generator_type: "synthetic"
  num_input_tokens: 50
  num_output_tokens: 20
  request_rate: 2.0
metrics_config:
  output_dir: "veeksha_baseline_output"
EOF

# Run benchmark
python -m veeksha.benchmark --from_file test_config_baseline.yaml
```

**Expected behavior**:
- Completes 5 requests
- Takes ~2.5 seconds (5 requests ÷ 2 req/s)
- No Revati messages in logs
- Metrics saved to `veeksha_baseline_output/`

---

### Test 5: Veeksha with Revati (Dispatch Only)

**Objective**: Test dispatch client integration without event loop integration

**Setup**:
```bash
# Ensure server is running
pkill -9 -f revati-server
sleep 2
revati-server -v &
sleep 2
```

**Config** (`test_config_revati_dispatch.yaml`):
```yaml
seed: 42
max_completed_requests: 5
api_url: "http://localhost:30000/v1"
api_key: "test-key"
enable_revati_client: true
revati_server_address: "tcp://localhost:5555"
client_config:
  model: "meta-llama/Llama-3.1-8B-Instruct"
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"
  llm_api: "openai_completions"
  num_clients: 1
  num_concurrent_requests_per_client: 2
request_generator_config:
  request_generator_type: "synthetic"
  num_input_tokens: 50
  num_output_tokens: 20
  request_rate: 2.0
metrics_config:
  output_dir: "veeksha_revati_dispatch_output"
```

**Command**:
```bash
# Run WITHOUT revati-run wrapper (dispatch client only)
python -m veeksha.benchmark --from_file test_config_revati_dispatch.yaml
```

**Expected logs**:
```
Initializing Revati client at tcp://localhost:5555 for dispatch timing
Revati dispatch client initialized and simulation started
Dispatched request ...
...
Finalizing and disconnecting Revati dispatch client
Revati dispatch client cleaned up successfully
```

**Expected behavior**:
- Dispatch client registers as "Dispatch"
- Time jumps control request dispatch timing
- Worker processes still use wall-clock time (aiohttp not intercepted)
- **Will likely hang** because workers aren't coordinated

**This test validates**:
- ✓ Client initialization
- ✓ Connection to server
- ✓ Finalization and cleanup
- ✗ Full coordination (need event loop integration)

---

### Test 6: Full Revati Integration (with Event Loop Integration)

**Objective**: Complete integration test with worker process coordination

**Setup**:
```bash
# Clean restart server
pkill -9 -f revati-server
sleep 2
revati-server -v 2>&1 | tee /tmp/revati_server_full.log &
sleep 2
```

**Config** (`test_config_revati_full.yaml`):
```yaml
seed: 42
max_completed_requests: 10
api_url: "http://localhost:30000/v1"
api_key: "test-key"
enable_revati_client: true
revati_server_address: "tcp://localhost:5555"
client_config:
  model: "meta-llama/Llama-3.1-8B-Instruct"
  tokenizer: "meta-llama/Llama-3.1-8B-Instruct"
  llm_api: "openai_completions"
  num_clients: 2
  num_concurrent_requests_per_client: 4
request_generator_config:
  request_generator_type: "synthetic"
  num_input_tokens: 100
  num_output_tokens: 50
  request_rate: 5.0
metrics_config:
  output_dir: "veeksha_revati_full_output"
```

**Command**:
```bash
# Run WITH revati-run wrapper (full integration)
revati-run --enable-event-loop-integration \
  python -m veeksha.benchmark --from_file test_config_revati_full.yaml
```

**Expected logs**:
```
Initializing Revati client at tcp://localhost:5555 for dispatch timing
Revati dispatch client initialized and simulation started
Starting async worker 0
Starting async worker 1
Dispatched request 0
Dispatched request 1
...
Finalizing and disconnecting Revati dispatch client
Revati dispatch client cleaned up successfully
Main loop completed.
```

**Server logs** (check `/tmp/revati_server_full.log`):
```
Client registered: Dispatch
Client registered: asyncio_loop_<pid>_<fd>
Client registered: asyncio_loop_<pid>_<fd>
...
Simulation started by Dispatch with N registered clients
Time advanced to X.XXX
...
Client finalized: Dispatch
```

**Verification**:
1. **Client count**: Should see Dispatch + (num_clients × num_concurrent_requests_per_client) worker clients
2. **Time advancement**: Should advance in virtual time only
3. **Completion**: All requests complete successfully
4. **Metrics**: Output saved to `veeksha_revati_full_output/`

**Performance comparison**:
```bash
# Compare with baseline
ls -lh veeksha_baseline_output/
ls -lh veeksha_revati_full_output/

# Check metrics
cat veeksha_baseline_output/summary.json
cat veeksha_revati_full_output/summary.json
```

---

### Test 7: Chrome Trace Visualization

**Objective**: Generate timeline visualization of benchmark execution

**Setup**:
```bash
pkill -9 -f revati-server
sleep 2

# Enable chrome trace generation
export REVATI_CHROME_TRACE=1
export REVATI_CHROME_TRACE_DIR=/tmp/veeksha_traces

revati-server -v &
sleep 2
```

**Command**:
```bash
revati-run --enable-event-loop-integration \
  python -m veeksha.benchmark --from_file test_config_revati_full.yaml
```

**Verify trace generation**:
```bash
ls -lh /tmp/veeksha_traces/
# Should see: chrome_trace_<timestamp>.json
```

**View trace**:
1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Click "Load"
4. Select the generated trace file

**What to look for in trace**:
- Timeline of request dispatch events
- Worker event loop activity
- Time jump operations
- Request/response timing

---

### Test 8: Error Handling and Fallback

**Objective**: Verify graceful fallback when Revati fails

**Test 8a: Server Not Running**

```bash
# Make sure server is NOT running
pkill -9 -f revati-server

# Try to run with revati enabled
python -m veeksha.benchmark \
  --from_file test_config_revati_full.yaml
```

**Expected behavior**:
- Warning logged: "Failed to initialize Revati client"
- Benchmark continues with normal time.sleep()
- Completes successfully

**Test 8b: Invalid Server Address**

```yaml
# Config with invalid address
enable_revati_client: true
revati_server_address: "tcp://invalid-host:9999"
```

```bash
python -m veeksha.benchmark --from_file test_config_invalid_address.yaml
```

**Expected behavior**:
- Connection timeout or error
- Warning logged
- Fallback to normal timing

---

### Test 9: Multi-Client Coordination

**Objective**: Verify dispatch client coordinates with worker event loops

**Test procedure**:
1. Start server with debug logging
2. Run benchmark with multiple workers
3. Monitor server logs for coordination

**Command**:
```bash
# Terminal 1: Server with debug logs
REVATI_LOGGING_LEVEL=DEBUG revati-server 2>&1 | tee /tmp/server_debug.log

# Terminal 2: Benchmark
revati-run --enable-event-loop-integration \
  python -m veeksha.benchmark \
  --enable_revati_client \
  --max_completed_requests 20 \
  --client_config.num_clients 2 \
  --client_config.num_concurrent_requests_per_client 4 \
  --request_generator_config.request_rate 10.0
```

**Verification in server logs**:
```bash
grep "Client registered" /tmp/server_debug.log | wc -l
# Should see: 1 (Dispatch) + 2*4 (workers) = 9 clients

grep "Time advanced" /tmp/server_debug.log | head -20
# Should see coordinated time advancement

grep "Client finalized" /tmp/server_debug.log
# Should see Dispatch finalized before workers
```

---

### Test 10: Stress Test

**Objective**: Test with high request rate and many workers

**Config** (`test_config_stress.yaml`):
```yaml
seed: 42
max_completed_requests: 100
enable_revati_client: true
client_config:
  num_clients: 4
  num_concurrent_requests_per_client: 8
request_generator_config:
  request_rate: 50.0  # High rate
  num_input_tokens: 200
  num_output_tokens: 100
```

**Command**:
```bash
pkill -9 -f revati-server
sleep 2
revati-server &
sleep 2

time revati-run --enable-event-loop-integration \
  python -m veeksha.benchmark --from_file test_config_stress.yaml
```

**Monitor**:
- System resources (CPU, memory)
- Server log size
- Client coordination latency
- Benchmark completion time

**Success criteria**:
- All 100 requests complete
- No deadlocks or hangs
- Metrics correctly collected
- Clean shutdown

---

## Debugging Guide

### Common Issues

#### Issue 1: Benchmark Hangs

**Symptoms**: Benchmark starts but never completes

**Debugging steps**:
```bash
# Check server logs
tail -f /tmp/revati_server.log

# Check which clients are registered
grep "Client registered" /tmp/revati_server.log

# Check time advancement
grep "Time advanced" /tmp/revati_server.log

# Check if any client is stuck
grep "time_jump" /tmp/revati_server.log | tail -20
```

**Common causes**:
- Dispatch client not finalized → Workers waiting forever
- Worker client crashed without unregistering
- Event loop not properly integrated

**Solutions**:
- Ensure finalize() called in cleanup
- Restart server between runs
- Check revati-run flags

#### Issue 2: "Client name already registered"

**Symptoms**: Server rejects client registration

**Cause**: Previous run didn't clean up

**Solution**:
```bash
# Always restart server between runs
pkill -9 -f revati-server
sleep 2
revati-server &
sleep 2
```

#### Issue 3: Workers Not Detected

**Symptoms**: Only Dispatch client registered, no worker clients

**Cause**: Not using revati-run wrapper

**Solution**:
```bash
# ✗ Wrong
python -m veeksha.benchmark ...

# ✓ Correct
revati-run --enable-event-loop-integration python -m veeksha.benchmark ...
```

#### Issue 4: Time Doesn't Advance

**Symptoms**: Virtual time stays at 0

**Debugging**:
```python
# Add debug logging to dispatch_requests
logger.info(f"Dispatch time_jump({sleep_time})")
new_time = revati_client.time_jump(sleep_time)
logger.info(f"Dispatch woke up at {new_time}")
```

**Common causes**:
- Client not properly registered
- Simulation not started
- Server coordination issue

---

## Validation Checklist

After integration, verify:

- [ ] Revati imports successfully (Test 1)
- [ ] CLI arguments recognized (Test 2)
- [ ] Dispatch client connects (Test 3)
- [ ] Baseline benchmark works (Test 4)
- [ ] Dispatch-only integration works (Test 5)
- [ ] Full integration with workers (Test 6)
- [ ] Chrome trace generation (Test 7)
- [ ] Error handling and fallback (Test 8)
- [ ] Multi-client coordination (Test 9)
- [ ] Stress test passes (Test 10)

---

## Performance Metrics

### Metrics to Collect

**Baseline (no Revati)**:
- Wall-clock duration
- Request throughput (req/s)
- Latency (TTFT, TPOT, E2E)

**With Revati**:
- Virtual time duration
- Wall-clock duration
- Speedup ratio (wall time / virtual time)
- Request throughput (virtual)
- Latency (virtual)

### Comparison Script

```python
import json

# Load baseline
with open('veeksha_baseline_output/summary.json') as f:
    baseline = json.load(f)

# Load revati run
with open('veeksha_revati_full_output/summary.json') as f:
    revati = json.load(f)

print(f"Baseline duration: {baseline['duration_s']:.2f}s")
print(f"Revati virtual duration: {revati['duration_s']:.2f}s")
print(f"Revati wall duration: {revati.get('wall_duration_s', 'N/A')}")

if 'wall_duration_s' in revati:
    speedup = baseline['duration_s'] / revati['wall_duration_s']
    print(f"Speedup: {speedup:.2f}x")
```

---

## Next Steps

After successful testing:

1. **Documentation**: Update main Veeksha README with Revati usage
2. **Examples**: Add example configs with Revati enabled
3. **CI/CD**: Add Revati integration tests to CI pipeline
4. **Metrics**: Implement wall time tracking for speedup metrics
5. **Analysis**: Create scripts to analyze chrome traces

---

## References

- Revati time sync docs: `/coc/scratch/docker_safe/kasra/revati/docs/usage/time_sync/`
- Veeksha architecture: `/coc/scratch/docker_safe/kasra/veeksha/README.md`
- SGLang integration example: `/coc/scratch/docker_safe/kasra/sglang/python/sglang/bench_serving.py`
