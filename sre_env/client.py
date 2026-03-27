from typing import Any, Dict
from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from .models import SREAction, SREObservation, SREState

class SREEnv(EnvClient[SREAction, SREObservation, SREState]):
    """Client for the SRE Autopilot environment."""

    def _step_payload(self, action: SREAction) -> dict:
        """Convert SREAction to JSON payload for step request."""
        return {"action": action.action, "service_id": action.service_id}

    def _parse_result(self, payload: dict) -> StepResult[SREObservation]:
        """Parse server response into StepResult[SREObservation]."""
        obs = SREObservation(**payload["observation"])
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SREState:
        """Parse server response into SREState object."""
        return SREState(**payload)

