"""E2E Test for Managed Server Benchmark."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import MagicMock, patch

import pytest
import yaml

from veeksha.new.benchmark import manage_benchmark_run
from veeksha.new.config.benchmark import BenchmarkConfig
from veeksha.new.config.utils import create_class_from_dict

SAMPLE_CONFIG_PATH = "veeksha/new/sample_configs/managed_server.yml"

class MockOpenAIHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len)
        data = json.loads(body)
        
        # Determine number of tokens requested
        max_tokens = data.get("max_completion_tokens", 10)
        
        response = {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": int(time.time()),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "mock " * max_tokens
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": max_tokens,
                "total_tokens": 10 + max_tokens
            }
        }
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode('utf-8'))
    
    def log_message(self, format, *args):
        pass # Silence logs

@pytest.fixture
def mock_openai_server():
    server = HTTPServer(('localhost', 0), MockOpenAIHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    yield server
    server.shutdown()

# @pytest.mark.e2e
def test_managed_server_benchmark(mock_openai_server, tmp_path) -> None:
    # 1. Load config
    with open(SAMPLE_CONFIG_PATH, "r") as f:
        config_dict = yaml.safe_load(f)
    
    # 2. Modify config for test
    config_dict["runtime"]["max_sessions"] = 5 
    config_dict["runtime"]["benchmark_timeout"] = 10
    config_dict["output_dir"] = str(tmp_path)
    
    # 3. Create BenchmarkConfig
    benchmark_config = create_class_from_dict(BenchmarkConfig, config_dict)
    
    # 4. Mock managed_server context manager
    port = mock_openai_server.server_port
    server_info = {
        "api_base": f"http://localhost:{port}/v1",
        "api_key": "dummy",
        "model": config_dict["server"]["model"]
    }
    
    mock_ctx = MagicMock()
    mock_ctx.__enter__.return_value = server_info
    mock_ctx.__exit__.return_value = None
    
    # We patch the call to managed_server in veeksha.new.benchmark
    with patch("veeksha.new.benchmark.managed_server", return_value=mock_ctx) as mocked_ms:
        # 5. Run
        result = manage_benchmark_run(benchmark_config)
        
        # 6. Verify
        # Check that mocked_ms was called
        mocked_ms.assert_called_once()
        
        # Check result
        # Metrics return e.g. "Time to First Chunk" from CDFSketch
        # The key names in result.metrics correspond to metric_name passed to CDFSketch
        
        # Looking at TextPerformanceEvaluator: "Time per Output Token"
        tpot_mean = result.metrics.get("Time per Output Token (Mean)")
        assert tpot_mean is not None
        assert tpot_mean >= 0
        
        ttfc_mean = result.metrics.get("Time to First Chunk (Mean)")
        assert ttfc_mean is not None
