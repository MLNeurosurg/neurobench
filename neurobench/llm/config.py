"""Load and validate judge-server settings, generation arguments, and execution limits."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_EXECUTION = {
    "max_concurrent": 16,
    "timeout_seconds": 180,
    "retry_handling": {
        "transport_error_retries": 5,
        "validation_error_retries": 5,
        "backoff_initial_seconds": 1,
        "backoff_max_seconds": 60,
    },
}
# Authentication is supplied separately and is never serialized into job files.
FORBIDDEN_ARGS = {
    "model",
    "messages",
    "stream",
    "api_key",
    "headers",
    "extra_headers",
    "authorization",
}


def check_request_args(value: dict) -> dict:
    """Return a copy of judge generation settings, rejecting credentials and reserved fields."""
    if not isinstance(value, dict):
        raise ValueError("request_args must be a JSON object")
    if FORBIDDEN_ARGS.intersection(k.lower() for k in value):
        raise ValueError(
            "request_args cannot override model/messages/stream or contain credentials/headers"
        )
    if "extra_body" in value:
        raise ValueError(
            "Put server-specific request fields directly in request_args, without extra_body"
        )
    if "seed" in value and (type(value["seed"]) is not int or value["seed"] < 0):
        raise ValueError("seed must be a nonnegative integer; omit it if unsupported")
    if "max_tokens" in value and "max_completion_tokens" in value:
        raise ValueError("Configure only one of max_tokens and max_completion_tokens")
    json.dumps(value, allow_nan=False)
    return deepcopy(value)


def resolve_config(config: dict) -> dict:
    """Validate judge settings and fill in default concurrency, timeout, and retry limits."""
    if not isinstance(config, dict) or set(config) - {
        "schema_version",
        "api",
        "execution",
    }:
        raise ValueError("Config accepts only schema_version, api, and execution")
    if config.get("schema_version", 1) != 1:
        raise ValueError("Unsupported config schema_version")
    api = config.get("api", {})
    if not isinstance(api, dict) or set(api) - {"base_url", "model", "request_args"}:
        raise ValueError("api accepts only base_url, model, and request_args")
    url = api.get("base_url", "")
    if not isinstance(url, str):
        raise ValueError("api.base_url must be an HTTP(S) URL")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "api.base_url must be an HTTP(S) URL without credentials, query, or fragment"
        )
    model = api.get("model", "")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("api.model must name the model served by the endpoint")
    supplied = config.get("execution", {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_EXECUTION):
        raise ValueError("Unknown execution setting")
    execution = deepcopy(DEFAULT_EXECUTION)
    retry = supplied.get("retry_handling", {})
    if not isinstance(retry, dict) or set(retry) - set(execution["retry_handling"]):
        raise ValueError("Unknown retry_handling setting")
    execution.update({k: v for k, v in supplied.items() if k != "retry_handling"})
    execution["retry_handling"].update(retry)
    for label, value in [
        ("max_concurrent", execution["max_concurrent"]),
        *[
            (k, execution["retry_handling"][k])
            for k in ("transport_error_retries", "validation_error_retries")
        ],
    ]:
        minimum = 1 if label == "max_concurrent" else 0
        if type(value) is not int or value < minimum:
            raise ValueError(f"{label} must be an integer >= {minimum}")
    for label, value in [
        ("timeout_seconds", execution["timeout_seconds"]),
        *[
            (k, execution["retry_handling"][k])
            for k in ("backoff_initial_seconds", "backoff_max_seconds")
        ],
    ]:
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or value < 0
            or (label == "timeout_seconds" and value == 0)
        ):
            raise ValueError(
                f"{label} must be finite and {'positive' if label == 'timeout_seconds' else 'nonnegative'}"
            )
    return {
        "schema_version": 1,
        "api": {
            "base_url": url.rstrip("/"),
            "model": model,
            "request_args": check_request_args(api.get("request_args", {})),
        },
        "execution": execution,
    }


def load_config(path: str | Path) -> dict:
    """Read a judge configuration JSON file and resolve its validated settings."""
    return resolve_config(json.loads(Path(path).read_text(encoding="utf-8")))
