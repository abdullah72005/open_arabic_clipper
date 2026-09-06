"""Source-local entity evidence built solely from transcript text."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceEntityMemory:
    """Observed entity forms and their supporting source segment indexes."""

    occurrences: dict[str, tuple[int, ...]]

    def contains_exact(self, surface_form: str) -> bool:
        return surface_form in self.occurrences

    def supports_change(self, old: str, new: str, evidence_ids: tuple[int, ...]) -> bool:
        """Permit only a changed form observed repeatedly in the same source."""

        if old == new:
            return True
        observed = self.occurrences.get(new, ())
        return len(observed) >= 2 and set(evidence_ids).issubset(observed)

    def with_observed_nomination(
        self,
        surface_form: str,
        evidence_ids: tuple[int, ...],
        segments: Sequence[Mapping[str, object]],
    ) -> SourceEntityMemory:
        """Admit a provider form only when every cited raw segment contains it verbatim."""

        if not evidence_ids or any(
            index < 0
            or index >= len(segments)
            or surface_form
            not in str(segments[index].get("raw_text", segments[index].get("text", "")))
            for index in evidence_ids
        ):
            return self
        merged = dict(self.occurrences)
        merged[surface_form] = tuple(sorted(set((*merged.get(surface_form, ()), *evidence_ids))))
        return SourceEntityMemory(merged)


def build_entity_memory(segments: Sequence[Mapping[str, object]]) -> SourceEntityMemory:
    """Capture exact observed Latin/numeric forms and repeated Arabic spans."""

    forms: dict[str, list[int]] = {}
    for index, segment in enumerate(segments):
        text = str(segment.get("raw_text", segment.get("text", "")))
        for form in _entity_forms(text):
            forms.setdefault(form, []).append(index)
    return SourceEntityMemory({form: tuple(indexes) for form, indexes in forms.items()})


def _entity_forms(text: str) -> set[str]:
    words = text.split()
    forms = {text_part for text_part in _latin_or_number_runs(text)}
    for width in (2, 3, 4):
        forms.update(
            " ".join(words[start : start + width]) for start in range(len(words) - width + 1)
        )
    return {form for form in forms if form}


def _latin_or_number_runs(text: str) -> list[str]:
    runs: list[str] = []
    current: list[str] = []
    for word in text.split():
        if any(character.isascii() and character.isalpha() for character in word) or any(
            character.isdigit() for character in word
        ):
            current.append(word)
        elif current:
            runs.append(" ".join(current))
            current = []
    if current:
        runs.append(" ".join(current))
    return runs
