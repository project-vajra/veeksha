
from typing import Dict, Any
import json
import hashlib


def get_config_hash(config: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
