"""Shadow evaluation boundary that deliberately has no executor reference."""
from __future__ import annotations

from typing import Optional, Protocol

from ecopadg.logging_schema import DecisionJSONLLogger, DecisionLog
from ecopadg.mpc.types import ControlAction, MPCDecision
from ecopadg.telemetry import ControlState


class DecisionProvider(Protocol):
    def decide(self, state: ControlState) -> MPCDecision:
        ...


class ShadowRunner:
    """Evaluate and log proposals without exposing any actuation method."""

    def __init__(
        self,
        controller: DecisionProvider,
        logger: Optional[DecisionJSONLLogger] = None,
    ):
        self.controller = controller
        self.logger = logger

    def evaluate(
        self,
        state: ControlState,
        *,
        active_action: Optional[ControlAction] = None,
    ) -> MPCDecision:
        decision = self.controller.decide(state)
        if self.logger is not None:
            self.logger.write(
                DecisionLog.from_decision(
                    state,
                    decision,
                    shadow=True,
                    controller="mpc",
                    active_action=active_action,
                )
            )
        return decision

    run = evaluate

