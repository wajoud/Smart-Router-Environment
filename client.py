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

from .models import (
    FIBEntry,
    LinkState,
    PacketInfo,
    SmartRouterAction,
    SmartRouterObservation,
)


class SmartRouterEnv(
    EnvClient[SmartRouterAction, SmartRouterObservation, State]
):
    """
    Client for the Smart Router Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with SmartRouterEnv(base_url="http://localhost:8000") as env:
        ...     result = env.reset()
        ...     obs = result.observation
        ...     print(f"Agent router: {obs.agent_router}")
        ...     print(f"Packet: {obs.packet.src} -> {obs.packet.dst}")
        ...
        ...     # Forward to neighbour at index 0
        ...     result = env.step(SmartRouterAction(next_hop_index=0))
        ...     print(f"Reward: {result.reward}")
    """

    def _step_payload(self, action: SmartRouterAction) -> Dict:
        return {"next_hop_index": action.next_hop_index}

    def _parse_result(self, payload: Dict) -> StepResult[SmartRouterObservation]:
        obs_data = payload.get("observation", {})
        if isinstance(obs_data, dict) and "observation" in obs_data:
            obs_data = obs_data["observation"]

        # Parse nested packet
        pkt_raw = obs_data.get("packet", {})
        packet = PacketInfo(
            packet_id=pkt_raw.get("packet_id", ""),
            src=pkt_raw.get("src", ""),
            dst=pkt_raw.get("dst", ""),
            priority=pkt_raw.get("priority", 1),
            ttl=pkt_raw.get("ttl", 0),
            hops_taken=pkt_raw.get("hops_taken", 0),
            visited=pkt_raw.get("visited", []),
        )

        # Parse link states
        link_states = [
            LinkState(
                index=l.get("index", i),
                neighbor=l.get("neighbor", ""),
                latency_ms=l.get("latency_ms", 0.0),
                utilization=l.get("utilization", 0.0),
                queue_depth=l.get("queue_depth", 0),
                is_congested=l.get("is_congested", False),
            )
            for i, l in enumerate(obs_data.get("link_states", []))
        ]

        # Parse FIB
        fib = [
            FIBEntry(
                destination=f.get("destination", ""),
                next_hops=f.get("next_hops", []),
            )
            for f in obs_data.get("fib", [])
        ]

        observation = SmartRouterObservation(
            agent_router=obs_data.get("agent_router", "R2"),
            packet=packet,
            link_states=link_states,
            fib=fib,
            queue_size=obs_data.get("queue_size", 0),
            active_flow_count=obs_data.get("active_flow_count", 0),
            network_utilization=obs_data.get("network_utilization", 0.0),
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
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
