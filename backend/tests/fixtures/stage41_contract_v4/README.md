# Stage 4.1 producer-contract V4 fixtures

These fixtures are deliberate, bounded extractions from the read-only
`clipfactory_stage6_acceptance` investigation snapshot
`/tmp/oac-stage41-audit-e7e067f.json` (SHA-256
`259d3128301b130da8573676211a84e1c7fc92de6039d28577210ec4998d92da`, matching the
2026-10-20 producer/validator audit).

Each JSON file contains only:

- the saved candidate row (identity, disposition, scores, hooks, summaries and
  provenance/risk fields);
- the selected FINAL_CLIP refinement row (authentic transcript, ordered word
  timestamps, bounds, dialect, evidence and fingerprints);
- the saved Stage 4.0 analysis row (source moment, assessments, platform risk
  and identity);
- the saved current strategy rows;
- the complete ordered authoritative source segment array (start/end/text per
  segment), because `hook_index`/`payoff_index` are positions in that array.

Arabic text, punctuation, numbers, protected tokens, ordering, bounds and
identities are preserved verbatim. No provider reply is contained here: the
authored provider-format responses used by the tests are constructed in
`tests/stage41_contract_v4_support.py` and are clearly labelled offline
fixtures. The extraction never mutates or depends on the application database.
