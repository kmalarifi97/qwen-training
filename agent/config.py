"""Agent configuration — env vars or config.json."""

import json
import os
from pathlib import Path


def _defaults() -> dict:
    return {
        "server_url": "ws://localhost:8000/agents/connect",
        "agent_name": "colab-gpu",
        "api_key": "",
        "heartbeat_interval_seconds": 15,
        "model_cache_dir": "/content/models",
        "log_dir": "/content/logs",
    }


def load_config() -> dict:
    config = _defaults()

    # Load from config.json next to this file
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config.update(json.load(f))

    # Env var overrides
    env_map = {
        "AGENT_SERVER_URL": "server_url",
        "AGENT_NAME": "agent_name",
        "AGENT_API_KEY": "api_key",
        "AGENT_HEARTBEAT": ("heartbeat_interval_seconds", int),
        "AGENT_MODEL_DIR": "model_cache_dir",
        "AGENT_LOG_DIR": "log_dir",
    }
    for env_key, target in env_map.items():
        val = os.environ.get(env_key)
        if val is not None:
            if isinstance(target, tuple):
                config[target[0]] = target[1](val)
            else:
                config[target] = val

    return config
