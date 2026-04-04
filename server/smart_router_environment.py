from uuid import uuid4
from pydantic import BaseModel
from typing import Any, Dict

# Meta OpenEnv Base Imports
from openenv.core.env_server import Environment
from openenv.core.env_server.types import State
from models import SmartRouterAction, SmartRouterObservation

# A global counter to ensure chaos triggers even if the class re-instantiates
TOTAL_STEPS = 0

try:
    from openenv.core.models import StepResult
except ImportError:
    from pydantic import BaseModel

    class StepResult(BaseModel):
        observation: Any
        reward: float
        done: bool
        info: Dict[str, Any] = {}


class SmartRouterEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> SmartRouterObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        return SmartRouterObservation(
            latency_ms=10.0, packet_loss=0.0, is_congested=False
        )

    def step(self, action: SmartRouterAction) -> StepResult:
        global TOTAL_STEPS
        TOTAL_STEPS += 1
        self._state.step_count += 1

        # Trigger chaos based on the global persistent counter
        chaos = TOTAL_STEPS % 5 == 0

        if action.path_selection == 0:  # Fiber
            lat, loss = (90.0, 0.10) if chaos else (10.0, 0.01)
        elif action.path_selection == 1:  # Copper
            lat, loss = 45.0, 0.02
        else:
            lat, loss = 500.0, 0.00

        reward_val = (100.0 / (lat + 1)) - (loss * 50)
        obs = SmartRouterObservation(
            latency_ms=lat, packet_loss=loss, is_congested=(lat > 60)
        )

        return StepResult(
            observation=obs,
            reward=float(reward_val),
            done=self._state.step_count >= 100,
            info={"chaos_active": chaos, "global_step": TOTAL_STEPS},
        )

    @property
    def state(self) -> State:
        return self._state
