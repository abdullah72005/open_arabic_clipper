"""Stage 4.3 deterministic final-plan selection.

Synchronous, transaction-safe, provider-free: bounded arbitration over at most
three persisted Stage 4.1 plans governed by Stage 4.2. It never replans,
re-governs, rewrites, researches, or renders.
"""

from app.transformation.selection.handoff import build_execution_handoff
from app.transformation.selection.policy import (
    SELECTION_FINGERPRINT_VERSION,
    SELECTION_POLICY_VERSION,
    SELECTION_SCHEMA_VERSION,
)
from app.transformation.selection.service import (
    SelectionView,
    evaluate_without_persisting,
    get_current_selection,
    get_selection,
    list_selections,
    read_selection,
    select_transformation_plan,
)

__all__ = [
    "SELECTION_FINGERPRINT_VERSION",
    "SELECTION_POLICY_VERSION",
    "SELECTION_SCHEMA_VERSION",
    "SelectionView",
    "build_execution_handoff",
    "evaluate_without_persisting",
    "get_current_selection",
    "get_selection",
    "list_selections",
    "read_selection",
    "select_transformation_plan",
]
