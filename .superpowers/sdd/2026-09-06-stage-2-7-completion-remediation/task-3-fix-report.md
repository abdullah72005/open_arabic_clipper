# Task 3 retry cleanup report

## RED

Before the repair, `ContextualReconstructor.reconstruct()` called the provider's
`release()` directly from `finally`. A cleanup exception therefore replaced a
successful `ReconstructionResult` (and there was no metadata field to record
the bounded warning). The release-failure regression test captured this failure
mode.

## GREEN

The cleanup call is now best effort. A release exception cannot replace valid
or provider-fallback output; it adds only the constant, non-sensitive metadata
`{"release_warning": "provider_release_failed"}`. `ReconstructionResult`
defaults metadata to an empty mapping for existing callers.

Verification in the backend container:

```text
19 passed, 1 warning in 2.68s
All checks passed!
```

Ruff passed for the touched service, result type, and reconstruction tests.
