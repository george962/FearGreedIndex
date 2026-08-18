"""Small, dependency-free HTTP retry helper for the CNN data jobs."""

from __future__ import annotations

import json
import random
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


def fetch_json_with_retry(
    request: Request,
    *,
    timeout: int,
    context: Any,
    attempts: int = 4,
    base_delay_seconds: float = 0.5,
    opener: Callable[..., Any] = urlopen,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch a JSON object with bounded exponential retry and jitter."""
    if attempts < 1:
        raise ValueError("attempts must be at least one")

    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with opener(request, timeout=timeout, context=context) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise RuntimeError("HTTP endpoint returned a non-object JSON payload")
            return payload
        except HTTPError as error:
            last_error = error
            if error.code not in RETRYABLE_HTTP_CODES or attempt == attempts - 1:
                raise RuntimeError(
                    f"HTTP endpoint returned {error.code} ({error.reason})"
                ) from error
        except URLError as error:
            last_error = error
            if attempt == attempts - 1:
                raise RuntimeError(f"Could not reach HTTP endpoint: {error.reason}") from error
        except json.JSONDecodeError as error:
            raise RuntimeError("HTTP endpoint returned invalid JSON") from error

        delay = base_delay_seconds * (2**attempt)
        sleep(delay + random.uniform(0.0, delay * 0.2))

    raise RuntimeError(f"HTTP request failed: {last_error}")
