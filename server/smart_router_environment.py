from uuid import uuid4
from pydantic import BaseModel
from typing import Any, Dict
import random

# Meta OpenEnv Base Imports
from openenv.core.env_server import Environment
from openenv.core.env_server.types import State
from models import SmartRouterAction, SmartRouterObservation

try:
    from .curriculum import CurriculumController
except ImportError:
    from server.curriculum import CurriculumController

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
        self.curriculum = CurriculumController()

        # Curriculum-controlled parameters
        self._chaos_interval = 5
        self._max_steps = 100

        # Episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0
        self._last_action = None
        self._reward_history = []

    def reset(self) -> SmartRouterObservation:
        # Get difficulty from curriculum
        difficulty = self.curriculum.get_difficulty()

        # Curriculum-controlled parameters
        # Easier = longer chaos intervals (more predictable)
        # Harder = shorter chaos intervals (more frequent)
        self._chaos_interval = max(3, int(10 - difficulty * 7))  # 10→3 steps

        # Easier = fewer steps per episode
        # Harder = more steps per episode
        self._max_steps = min(100, int(20 + difficulty * 80))  # 20→100 steps

        # Reset episode state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_step = 0
        self._episode_reward = 0.0
        self._last_action = None
        self._reward_history = []

        return SmartRouterObservation(
            latency_ms=10.0, packet_loss=0.0, is_congested=False
        )

    def step(self, action: SmartRouterAction) -> StepResult:
        self._episode_step += 1
        self._state.step_count += 1

        # Trigger chaos based on episode step (not global counter)
        chaos = self._episode_step % self._chaos_interval == 0

        # Get difficulty for potential noise/randomness
        difficulty = self.curriculum.get_difficulty()

        # Base latency and packet loss by path
        if action.path_selection == 0:  # Fiber
            lat, loss = (90.0, 0.10) if chaos else (10.0, 0.01)
        elif action.path_selection == 1:  # Copper
            lat, loss = 45.0, 0.02
        else:  # Satellite
            lat, loss = 500.0, 0.00

        # Add latency noise that increases with difficulty
        if difficulty > 0.5:
            noise_std = (difficulty - 0.5) * 10  # 0ms → 5ms std dev at max difficulty
            lat += random.gauss(0, noise_std)
            lat = max(1.0, lat)  # Ensure positive latency

        # Base reward: latency and packet loss
        base_reward = (100.0 / (lat + 1)) - (loss * 50)

        # Reward shaping: penalize switching (simulates BGP convergence cost)
        switching_penalty = 0.0
        if self._last_action is not None and action.path_selection != self._last_action:
            switching_penalty = 0.5

        # Reward shaping: bonus for anticipating chaos
        anticipation_bonus = 0.0
        if self._episode_step == (self._chaos_interval - 1) and action.path_selection == 1:
            # Agent switched to Copper before chaos hits
            anticipation_bonus = 1.0

        # Reward shaping: consistency bonus (low variance in recent rewards)
        consistency_bonus = 0.0
        if len(self._reward_history) >= 10:
            import statistics
            recent_std = statistics.stdev(self._reward_history[-10:])
            if recent_std < 2.0:  # Stable performance
                consistency_bonus = 0.3

        # Total reward with shaping
        reward_val = base_reward - switching_penalty + anticipation_bonus + consistency_bonus

        # Track for episode statistics
        self._episode_reward += reward_val
        self._reward_history.append(base_reward)
        self._last_action = action.path_selection

        obs = SmartRouterObservation(
            latency_ms=lat, packet_loss=loss, is_congested=(lat > 60)
        )

        done = self._state.step_count >= self._max_steps

        # Record episode result for curriculum when done
        if done:
            # Define success threshold based on optimal strategy estimate
            # Optimal should get ~7-8 avg reward per step
            success_threshold = 5.0 * self._max_steps
            success = self._episode_reward >= success_threshold

            self.curriculum.record(
                task_type="routing",
                success=success,
                steps=self._episode_step,
                reward=self._episode_reward
            )

        return StepResult(
            observation=obs,
            reward=float(reward_val),
            done=done,
            info={
                "chaos_active": chaos,
                "episode_step": self._episode_step,
                "chaos_interval": self._chaos_interval,
                "max_steps": self._max_steps,
                "difficulty": difficulty,
                "curriculum_tier": self.curriculum.get_tier_name(),
                "switching_penalty": switching_penalty,
                "anticipation_bonus": anticipation_bonus,
                "consistency_bonus": consistency_bonus,
                "episode_reward": self._episode_reward,
            },
        )

    @property
    def state(self) -> State:
        return self._state
