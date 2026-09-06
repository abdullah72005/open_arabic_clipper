# Task 4 contract continuation

## RED (regression tests first)

Command:

```text
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c "python -m pip install pytest >/dev/null && PYTHONPATH=/app pytest tests/test_task4_contract_remainder.py -q"
```

Result: collection failed with `ImportError: cannot import name 'batch_generation_requests' from app.transcription.reconstruction.providers`.

## GREEN

Command:

```text
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c "python -m pip install pytest pytest-asyncio >/dev/null && PYTHONPATH=/app pytest tests/test_reconstruction_*.py -q"
```

Result: `46 passed in 3.33s`.

## Ruff

Command:

```text
docker compose run --rm --no-deps -v "$(pwd)/backend:/app" backend sh -c "python -m pip install ruff >/dev/null && ruff check app/transcription/reconstruction/providers.py app/transcription/reconstruction/routing.py app/transcription/reconstruction/service.py app/transcription/reconstruction/ollama.py app/core/settings.py tests/test_task4_contract_remainder.py"
```

Result: `All checks passed!`.
