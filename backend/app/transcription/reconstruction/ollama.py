"""Managed Ollama lifecycle around the OpenAI-compatible inference protocol."""

from app.transcription.reconstruction.providers import (
    HttpRequest,
    OpenAICompatibleReconstructionProvider,
    ProviderResponseError,
)
from app.transcription.reconstruction.types import ProviderAvailability, ProviderHealth


class OllamaReconstructionProvider(OpenAICompatibleReconstructionProvider):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_seconds: float,
        release_after_run: bool = True,
        request: HttpRequest | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            request=request,
        )
        self.provider_name = "ollama"
        self.release_after_run = release_after_run

    def health(self) -> ProviderHealth:
        try:
            payload = self._json_request("GET", "/api/tags", None)
            models = payload.get("models")
            if not isinstance(models, list):
                raise ProviderResponseError("provider response is missing models")
        except ProviderResponseError:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "ollama",
                self.model,
                None,
                "provider health check failed",
            )
        match = next(
            (item for item in models if isinstance(item, dict) and item.get("name") == self.model),
            None,
        )
        if match is None:
            return ProviderHealth(
                ProviderAvailability.UNAVAILABLE,
                "ollama",
                self.model,
                None,
                f"configured model {self.model} is not installed",
            )
        return ProviderHealth(
            ProviderAvailability.AVAILABLE,
            "ollama",
            self.model,
            str(match.get("digest") or "") or None,
            "model available",
        )

    def release(self) -> None:
        if self.release_after_run:
            self._json_request(
                "POST",
                "/api/generate",
                {"model": self.model, "keep_alive": 0},
            )
