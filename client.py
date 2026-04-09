# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Smart Router Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SmartRouterAction, SmartRouterObservation


class SmartRouterEnv(
    EnvClient[SmartRouterAction, SmartRouterObservation, State]
):
    """
    Client for the Smart Router Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SmartRouterEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(f"Latency: {result.observation.latency_ms}ms")
        ...
        ...     result = client.step(SmartRouterAction(path_selection=0))
        ...     print(f"Reward: {result.reward}")

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SmartRouterEnv.from_docker_image("smart_router-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SmartRouterAction(path_selection=0))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SmartRouterAction) -> Dict:
        """
        Convert SmartRouterAction to JSON payload for step message.

        Args:
            action: SmartRouterAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "path_selection": action.path_selection,
        }

    def _parse_result(self, payload: Dict) -> StepResult[SmartRouterObservation]:
        """
        Parse server response into StepResult[SmartRouterObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SmartRouterObservation
        """
        obs_data = payload.get("observation", {})
        observation = SmartRouterObservation(
            latency_ms=obs_data.get("latency_ms", 0.0),
            packet_loss=obs_data.get("packet_loss", 0.0),
            is_congested=obs_data.get("is_congested", False),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
