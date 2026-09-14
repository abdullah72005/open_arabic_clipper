"""Explicit local (Qwen/Ollama) Stage 4.0 strategy-discovery adapter.

Only used for ``local_only`` mode with ``CLIPFACTORY_LOCAL_QWEN_ENABLED=true``.
The provider performs no lease handling; the executor wraps actual local
inference with the shared heavy-model lease. Qwen is text-only and can never
claim audio recovery or factual verification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from typing import cast
from urllib.request import Request, urlopen

from app.candidates.providers import ProviderErrorCategory
from app.transcription.reconstruction.providers import (
    ProviderResponseError,
    _extract_json_object,
)
from app.transformation.policy import SCHEMA_VERSION
from app.transformation.providers import (
    TRANSFORMATION_SYSTEM_INSTRUCTION,
    TransformationProviderError,
    TransformationStrategyRequest,
    parse_strategy_results,
    transformation_prompt_hash,
)
from app.transformation.types import TransformationProviderResult

HttpRequest = Callable[[str, str, bytes | None, dict[str, str], float], bytes]


class LocalTransformationProvider:
    provider_name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float = 180.0,
        max_output_tokens: int = 1_024,
        temperature: float = 0.0,
        request: HttpRequest | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self.model = model
        self._timeout = max(0.1, float(timeout_seconds))
        self._output_tokens = max(1, int(max_output_tokens))
        self._temperature = max(0.0, float(temperature))
        self._request = request or _request_bytes
        self._digest = hashlib.sha256(f"ollama-transformation:{model}".encode()).hexdigest()
        self.rate_limited = False
        self._usage: dict[str, int] = {
            "prompt_token_count": 0,
            "candidates_token_count": 0,
            "total_token_count": 0,
            "thoughts_token_count": 0,
        }

    def release(self) -> None:
        return None

    def runtime_identity(self) -> dict[str, object]:
        return {
            "provider": "ollama",
            "model": self.model,
            "digest": self._digest,
            "prompt_hash": transformation_prompt_hash(),
            "schema_version": SCHEMA_VERSION,
            "temperature": self._temperature,
            "max_output_tokens": self._output_tokens,
        }

    def refresh_runtime_identity(self) -> dict[str, object]:
        return self.runtime_identity()

    def usage_summary(self) -> dict[str, int]:
        return dict(self._usage)

    def discover(
        self, requests: Sequence[TransformationStrategyRequest]
    ) -> dict[str, TransformationProviderResult]:
        if not requests:
            return {}
        payload = {"candidates": [request.to_payload() for request in requests]}
        instruction = TRANSFORMATION_SYSTEM_INSTRUCTION
        if self.model.startswith("qwen3"):
            instruction = instruction + " /no_think"
        body = {
            "model": self.model,
            "temperature": self._temperature,
            "max_tokens": self._output_tokens,
            "reasoning_effort": "none",
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        }
        content = self._call(body)
        return parse_strategy_results(content, requests)

    def _call(self, body: dict[str, object]) -> dict[str, object]:
        encoded = json.dumps(body, ensure_ascii=False).encode()
        try:
            response = self._request(
                "POST",
                f"{self._base_url}/v1/chat/completions",
                encoded,
                {"Content-Type": "application/json"},
                self._timeout,
            )
            parsed = json.loads(response.decode())
            choices = parsed["choices"]
            message = choices[0]["message"]
            text = message["content"]
            if not isinstance(text, str):
                raise TypeError
            result = _extract_json_object(text)
        except ProviderResponseError as error:
            raise TransformationProviderError(
                ProviderErrorCategory.MALFORMED_OUTPUT.value, "local transformation output invalid"
            ) from error
        except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError, OSError):
            raise TransformationProviderError(
                ProviderErrorCategory.PROVIDER_ERROR.value, "local transformation request failed"
            ) from None
        if not isinstance(result, dict):
            raise TransformationProviderError(ProviderErrorCategory.MALFORMED_OUTPUT.value)
        return result


def _request_bytes(
    method: str, url: str, body: bytes | None, headers: dict[str, str], timeout: float
) -> bytes:
    with urlopen(
        Request(url, data=body, headers=headers, method=method), timeout=timeout
    ) as response:  # noqa: S310
        return cast(bytes, response.read())
