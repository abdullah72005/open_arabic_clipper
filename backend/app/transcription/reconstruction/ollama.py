"""Managed Ollama lifecycle around the OpenAI-compatible inference protocol."""

import time
from collections.abc import Callable

from app.transcription.reconstruction.providers import (
    HttpRequest,
    ModelNotFoundError,
    OpenAICompatibleReconstructionProvider,
    ProviderResponseError,
)
from app.transcription.reconstruction.types import (
    ProviderAvailability,
    ProviderHealth,
    UnloadOutcome,
)


class OllamaReconstructionProvider(OpenAICompatibleReconstructionProvider):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        release_after_run: bool = True,
        max_context_tokens: int | None = None,
        output_tokens: int = 256,
        chat_framing_reserve: int = 64,
        safety_reserve: int = 128,
        unload_timeout_seconds: float = 30.0,
        unload_poll_seconds: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        request: HttpRequest | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_context_tokens=max_context_tokens,
            output_tokens=output_tokens,
            chat_framing_reserve=chat_framing_reserve,
            safety_reserve=safety_reserve,
            request=request,
        )
        self.provider_name = "ollama"
        self.release_after_run = release_after_run
        self._unload_timeout = unload_timeout_seconds
        self._unload_poll = unload_poll_seconds
        self._sleep = sleep
        self._monotonic = monotonic

    def _fetch_live_digest(self) -> str | None:
        payload = self._json_request("GET", "/api/tags", None)
        models = payload.get("models")
        if not isinstance(models, list):
            raise ProviderResponseError("provider response is missing models")
        match = next(
            (item for item in models if isinstance(item, dict) and item.get("name") == self.model),
            None,
        )
        if match is None:
            raise ModelNotFoundError(f"configured model {self.model} is not installed")
        return str(match.get("digest") or "") or None

    def health(self) -> ProviderHealth:
        try:
            digest = self._fetch_live_digest()
        except ModelNotFoundError as error:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "ollama",
                self.model,
                None,
                str(error),
            )
        except ProviderResponseError:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "ollama",
                self.model,
                None,
                "provider health check failed",
            )
        self._model_digest = digest
        return ProviderHealth(
            ProviderAvailability.AVAILABLE,
            "ollama",
            self.model,
            digest,
            "model available",
        )

    def release(self) -> UnloadOutcome | None:
        """Unload the model and poll until it disappears or the timeout warns."""

        if not self.release_after_run:
            return None
        started = self._monotonic()
        try:
            self._json_request(
                "POST",
                "/api/generate",
                {"model": self.model, "keep_alive": 0},
            )
        except (OSError, ProviderResponseError):
            return UnloadOutcome(False, False, 0.0, "unload request failed")
        confirmed = False
        while self._monotonic() - started < self._unload_timeout:
            if not self._model_resident():
                confirmed = True
                break
            self._sleep(self._unload_poll)
        warning = None if confirmed else "model still resident after unload timeout"
        return UnloadOutcome(True, confirmed, self._monotonic() - started, warning)

    def _model_resident(self) -> bool:
        """Poll the process listing; failures are conservative (assume resident)."""

        try:
            payload = self._json_request("GET", "/api/ps", None)
        except (OSError, ProviderResponseError):
            return True
        models = payload.get("models")
        if not isinstance(models, list):
            return True
        return any(
            isinstance(item, dict)
            and (
                item.get("name") == self.model
                or str(item.get("name", "")).startswith(f"{self.model}:")
            )
            for item in models
        )
