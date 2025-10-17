# Veeksha Dashboard 

## Enabling the Dashboard

### Single Benchmark

Add the following to any benchmark config:

```yaml
dashboard_config:
  enabled: true
  # ...
```

The dashboard will automatically start at `http://localhost:5000` and remain open until you press Ctrl+C.

### Multiple Benchmarks

Currently dashboard launches if only one benchmark config has the dashboard enabled. This may need to be changed/dashboard config refactored out to somewhere else

## Accessing the Dashboard

### Local Access

If running on your local machine, open your browser to:
```
http://localhost:5000
```

### Remote Access (SSH)

When running benchmarks on a remote GPU server, you have two options:

**Option 1: VS Code Port Forwarding (Automatic)**

If you're using VS Code's Remote SSH extension, ports are automatically forwarded. Just open `http://localhost:5000` in your local browser.

**Option 2: Manual SSH Port Forwarding**

If automatic forwarding doesn't work or you're not using VS Code:

1. In a new terminal on your local machine, run:
   ```bash
   ssh -L 5001:localhost:5000 vmehrotra7@badger.cc.gatech.edu
   ```

   Note: Using local port 5001 because my macOS had something running on port 5000. Might add some flexibility by having the port be configurable in the dashboard config.

2. Keep this SSH session open

3. Open your browser to:
   ```bash
   http://localhost:5001
   ```

Note: these instructions are printed to console when the dashboard is enabled.