"""Policy boundary for automation introduced after Stage 1."""

from app.core.enums import RightsStatus


class AutopilotAuthorizationError(PermissionError):
    """Raised when an automated later-stage action lacks explicit rights."""


def require_autopilot_authorization(rights_status: RightsStatus) -> None:
    """Require declared ownership or authorization before later automation."""
    if rights_status is RightsStatus.UNKNOWN:
        raise AutopilotAuthorizationError(
            "AUTOPILOT processing is blocked while source rights are UNKNOWN"
        )
