"""Submit LLM judge requests with bounded concurrency, validation, retries, and saved audit records."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Callable

import httpx
from tqdm import tqdm

from neurobench import __version__
from .config import check_request_args, resolve_config
from neurobench.data.io import (
    append_jsonl,
    directory_lock,
    require_empty_directory,
    write_json,
    write_jsonl,
)

RETRYABLE = {408, 409, 425, 429}


@dataclass(frozen=True)
class Request:
    """Describe one judge request and any overrides to the shared generation settings."""

    id: str
    messages: list[dict[str, str]]
    request_args: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Validate and serialize a planned judge request for the requests audit."""
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Request IDs must be nonblank strings")
        if not isinstance(self.messages, list) or not self.messages:
            raise ValueError(f"Request {self.id}: messages must be a nonempty list")
        for message in self.messages:
            if (
                not isinstance(message, dict)
                or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or not isinstance(message["content"], str)
            ):
                raise ValueError(
                    f"Request {self.id}: expected text messages with role and content"
                )
        check_request_args(self.request_args)
        return asdict(self)


Validator = Callable[[Request, str], None]


class LocalLLMClient:
    """Submit judge requests concurrently with validated replies, retries, and saved audit records."""

    def __init__(self, config: dict, api_key_path: str | Path | None = None):
        """Resolve judge settings and load an optional API token kept outside saved configuration."""
        self.config = resolve_config(config)
        self._api_key = (
            Path(api_key_path).read_text(encoding="utf-8").strip()
            if api_key_path
            else None
        )
        if api_key_path and not self._api_key:
            raise ValueError("API key file is empty")

    def run(
        self,
        requests: list[Request],
        output_dir: str | Path,
        *,
        validator: Validator | None = None,
    ) -> list[dict]:
        """Run a judge job synchronously in a new or empty audit directory."""
        return asyncio.run(
            self.run_async(
                requests, output_dir, validator=validator,
            )
        )

    async def run_async(
        self,
        requests: list[Request],
        output_dir: str | Path,
        *,
        validator: Validator | None = None,
    ) -> list[dict]:
        """Run a judge job asynchronously and return final replies in request order."""
        require_empty_directory(output_dir)
        request_rows = [request.to_dict() for request in requests]
        ids = [request.id for request in requests]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate request IDs")
        for request in requests:
            check_request_args(
                self.config["api"]["request_args"] | request.request_args
            )
        with directory_lock(output_dir) as directory:
            require_empty_directory(directory)
            write_json(directory / 'config.json', {'code_version': __version__, 'config': self.config})
            write_jsonl(directory / 'requests.jsonl', request_rows)
            write_jsonl(directory / 'responses.jsonl', [])
            (directory / 'attempts.jsonl').touch()
            completed = {}
            semaphore = asyncio.Semaphore(self.config["execution"]["max_concurrent"])
            headers = (
                {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
            )
            # Keep judge requests on the configured route; this client owns retry handling.
            async with httpx.AsyncClient(
                headers=headers,
                timeout=self.config["execution"]["timeout_seconds"],
                trust_env=False,
                follow_redirects=False,
                limits=httpx.Limits(
                    max_connections=self.config["execution"]["max_concurrent"],
                    max_keepalive_connections=self.config["execution"][
                        "max_concurrent"
                    ],
                ),
            ) as session:
                queue = asyncio.Queue()
                for request in requests:
                    queue.put_nowait(request)

                # Progress counts resolved requests, including exhausted failures, not retry attempts.
                refresh_interval = 1.0 if sys.stderr.isatty() else 30.0
                with tqdm(
                    total=len(requests), initial=0, desc='Judge requests',
                    unit='request', dynamic_ncols=True, mininterval=refresh_interval,
                    postfix={'successful': 0, 'failed': 0},
                ) as progress:
                    counts = {'successful': 0, 'failed': 0}

                    async def refresh_progress():
                        """Refresh elapsed-time reporting while judge requests remain in flight."""
                        while True:
                            await asyncio.sleep(refresh_interval)
                            progress.refresh()

                    async def worker():
                        """Process queued judge requests and record each final reply and progress update."""
                        while not queue.empty():
                            request = queue.get_nowait()
                            result = await self._execute(
                                request, session, semaphore, directory, validator
                            )
                            completed[request.id] = result
                            append_jsonl(directory / 'responses.jsonl', result)
                            counts['successful' if result['status'] == 'success' else 'failed'] += 1
                            progress.set_postfix(counts, refresh=False)
                            progress.update(1)

                    workers = [
                        asyncio.create_task(worker())
                        for _ in range(
                            min(len(requests), self.config['execution']['max_concurrent'])
                        )
                    ]
                    heartbeat = asyncio.create_task(refresh_progress()) if requests else None
                    try:
                        await asyncio.gather(*workers)
                    finally:
                        tasks = workers + ([heartbeat] if heartbeat is not None else [])
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            results = [completed[request_id] for request_id in ids]
            write_jsonl(directory / "responses.jsonl", results)
            return results

    def _safe_error(self, value) -> str:
        """Remove the configured API token from errors before writing judge audit records."""
        text = str(value)
        return text.replace(self._api_key, "[REDACTED]") if self._api_key else text

    async def _send(self, session, payload):
        """Submit one HTTP chat-completion request to the configured judge server."""
        return await session.post(
            self.config["api"]["base_url"] + "/chat/completions", json=payload
        )

    async def _execute(self, request, session, semaphore, directory, validator):
        """Retry one judge request while preserving the vote's messages, seed, and generation settings."""
        retry = self.config["execution"]["retry_handling"]
        args = deepcopy(self.config["api"]["request_args"] | request.request_args)
        calls = 0
        for validation_attempt in range(1, retry["validation_error_retries"] + 2):
            current_args = deepcopy(args)
            # A retry is still the same vote: preserve its seed and decoding settings.
            payload = {
                "model": self.config["api"]["model"],
                "messages": request.messages,
                **current_args,
                "stream": False,
            }
            for transport_attempt in range(1, retry["transport_error_retries"] + 2):
                calls += 1
                output, reasoning, usage, finish_reason = "", "", {}, None
                error, status, retryable, http_status = (
                    "",
                    "failed_api_request",
                    False,
                    None,
                )
                try:
                    async with semaphore:
                        response = await self._send(session, payload)
                    http_status = response.status_code
                    if not 200 <= http_status < 300:
                        error = f"HTTP {http_status}: {response.text[:2000]}"
                        retryable = http_status in RETRYABLE or 500 <= http_status < 600
                    else:
                        status = "failed_validation"
                        try:
                            body = response.json()
                            choice = body["choices"][0]
                            message = choice["message"]
                            output = message.get("content") or ""
                            reasoning = (
                                message.get("reasoning_content")
                                or message.get("reasoning")
                                or message.get("thought")
                                or ""
                            )
                            usage = body.get("usage") or {}
                            finish_reason = choice.get("finish_reason")
                            if not isinstance(output, str) or not output.strip():
                                raise ValueError("Empty or non-text response content")
                            if finish_reason == "length":
                                raise ValueError(
                                    "Response truncated at the configured token limit"
                                )
                            if validator is not None:
                                validator(deepcopy(request), output)
                            status = "success"
                        except (
                            ValueError,
                            TypeError,
                            KeyError,
                            IndexError,
                            AttributeError,
                        ) as exc:
                            error = f"Invalid response: {exc}"
                except httpx.TransportError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    retryable = True
                result = {
                    "id": request.id,
                    "status": status,
                    "response": output,
                    "reasoning": reasoning,
                    "error": self._safe_error(error),
                    "usage": usage,
                    "finish_reason": finish_reason,
                    "http_status": http_status,
                    "validation_attempts": validation_attempt,
                    "api_calls": calls,
                }
                append_jsonl(
                    directory / "attempts.jsonl",
                    {
                        "request_id": request.id,
                        "time": datetime.now(timezone.utc).isoformat(),
                        "validation_attempt": validation_attempt,
                        "transport_attempt": transport_attempt,
                        "request_args": current_args,
                        "messages": request.messages,
                        "status": status,
                        "result": result,
                    },
                )
                if status == "success":
                    return result
                if status == "failed_validation":
                    break
                if (
                    not retryable
                    or transport_attempt > retry["transport_error_retries"]
                ):
                    return result
                delay = min(
                    retry["backoff_max_seconds"],
                    retry["backoff_initial_seconds"]
                    * 2 ** min(transport_attempt - 1, 30),
                )
                await asyncio.sleep(delay)
        return result
