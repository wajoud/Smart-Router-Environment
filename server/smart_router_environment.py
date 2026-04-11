"""
Smart Router RL Environment — multi-hop network simulation.

The agent IS router R2. Each step a packet arrives at R2 and the agent
must pick a next-hop (by index into link_states) to forward it toward its
destination. Background traffic loads links independently, creating dynamic
congestion the agent must learn to route around.
"""

from __future__ import annotations

import random
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server import Environment
from openenv.core.env_server.types import State

try:
    from models import (
        FIBEntry,
        LinkState,
        PacketInfo,
        SmartRouterAction,
        SmartRouterObservation,
    )
except ImportError:
    from smart_router.models import (  # type: ignore
        FIBEntry,
        LinkState,
        PacketInfo,
        SmartRouterAction,
        SmartRouterObservation,
    )

try:
    from .curriculum import CurriculumController
    from .topology import NetworkGraph, generate_topology
    from .traffic_engine import TrafficEngine
except ImportError:
    from server.curriculum import CurriculumController
    from server.topology import NetworkGraph, generate_topology
    from server.traffic_engine import TrafficEngine

try:
    from openenv.core.models import StepResult
except ImportError:
    from pydantic import BaseModel

    class StepResult(BaseModel):  # type: ignore
        observation: Any
        reward: float
        done: bool
        info: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Remainder-simulation result
# ---------------------------------------------------------------------------

@dataclass
class _RemainderResult:
    delivered: bool
    latency_ms: float
    hops: int
    drop_reason: str = ""


# ---------------------------------------------------------------------------
# Priority helpers
# ---------------------------------------------------------------------------

_PRIORITY_MULT = [0.5, 1.0, 2.0]      # low / medium / high
_PRIORITY_LABEL = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SmartRouterEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, task_type: str = "routing") -> None:
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.curriculum = CurriculumController()
        self.task_type = task_type

        # Curriculum-controlled episode parameters
        self._chaos_interval = 5
        self._max_steps = 100

        # Episode tracking
        self._episode_step = 0
        self._episode_reward = 0.0
        self._last_action: Optional[int] = None
        self._reward_history: List[float] = []

        # Multi-hop simulation state
        self._graph: Optional[NetworkGraph] = None
        self._traffic: Optional[TrafficEngine] = None
        self._pending_packets: Deque[PacketInfo] = deque()
        self._packets_delivered = 0
        self._packets_dropped = 0
        self._total_delivery_latency = 0.0

        # Precomputed hop distances (updated each reset)
        self._hop_dist: Dict[str, Dict[str, int]] = {}

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self) -> SmartRouterObservation:
        difficulty = self.curriculum.get_difficulty()

        # Regenerate topology scaled to current difficulty
        self._graph = generate_topology(difficulty)
        self._traffic = TrafficEngine(self._graph, difficulty)

        # Curriculum-controlled episode length
        self._max_steps = min(100, int(20 + difficulty * 80))

        # Reset all episode state
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._episode_step = 0
        self._episode_reward = 0.0
        self._last_action = None
        self._reward_history = []
        self._packets_delivered = 0
        self._packets_dropped = 0
        self._total_delivery_latency = 0.0
        self._pending_packets = deque()

        # Precompute BFS hop distances for progress reward
        self._hop_dist = self._bfs_all_distances()

        # Populate packet queue
        self._refill_packet_queue(difficulty)

        return self._build_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: SmartRouterAction) -> StepResult:
        self._episode_step += 1
        self._state.step_count += 1
        difficulty = self.curriculum.get_difficulty()

        # Dequeue next packet
        packet = self._pending_packets.popleft()
        neighbors = self._graph.neighbors(self._graph.agent_router)
        pmult = _PRIORITY_MULT[min(packet.priority, 2)]

        reward_val: float
        error_tag: str = ""
        info_extra: Dict[str, Any] = {}

        # ---- Validate action ------------------------------------------
        if action.next_hop_index == -1:
            # Intentional drop
            reward_val = -1.0 * pmult
            self._packets_dropped += 1
            error_tag = "INTENTIONAL_DROP"

        elif action.next_hop_index >= len(neighbors) or action.next_hop_index < 0:
            # Index out of range → invalid
            reward_val = -10.0 * pmult
            self._packets_dropped += 1
            error_tag = f"INVALID_INDEX:{action.next_hop_index}_max:{len(neighbors)-1}"

        else:
            next_hop = neighbors[action.next_hop_index]

            if next_hop in packet.visited:
                # Cycle detected
                reward_val = -8.0 * pmult
                self._packets_dropped += 1
                error_tag = f"CYCLE:{next_hop}_already_in_visited"

            elif packet.ttl <= 1:
                # Would expire on this hop
                reward_val = -5.0 * pmult
                self._packets_dropped += 1
                error_tag = "TTL_EXPIRED"

            else:
                # Valid forwarding decision
                link = (self._graph.agent_router, next_hop)
                link_lat = self._traffic.get_effective_latency(link)
                link_util = self._traffic.get_link_utilization(link)

                remainder = self._simulate_remainder(
                    from_node=next_hop,
                    dst=packet.dst,
                    ttl=packet.ttl - 1,
                    visited=list(packet.visited) + [self._graph.agent_router],
                )
                total_latency = link_lat + remainder.latency_ms

                reward_val = self._compute_reward(
                    packet=packet,
                    next_hop=next_hop,
                    link_lat=link_lat,
                    link_util=link_util,
                    total_latency=total_latency,
                    delivered=remainder.delivered,
                )

                if remainder.delivered:
                    self._packets_delivered += 1
                    self._total_delivery_latency += total_latency
                else:
                    self._packets_dropped += 1
                    if remainder.drop_reason:
                        error_tag = f"REMAINDER_FAIL:{remainder.drop_reason}"

                info_extra = {
                    "next_hop": next_hop,
                    "link_latency_ms": round(link_lat, 2),
                    "link_utilization": round(link_util, 3),
                    "total_latency_ms": round(total_latency, 2),
                    "delivered": remainder.delivered,
                    "fib_suggested": (
                        self._graph.fib.get(self._graph.agent_router, {})
                        .get(packet.dst, ["?"])[0]
                    ),
                }

        # ---- Reward shaping: consistency bonus (unchanged) ---------------
        consistency_bonus = 0.0
        if len(self._reward_history) >= 10:
            try:
                if statistics.stdev(self._reward_history[-10:]) < 2.0:
                    consistency_bonus = 0.3
            except statistics.StatisticsError:
                pass
        reward_val += consistency_bonus

        self._episode_reward += reward_val
        self._reward_history.append(reward_val)
        self._last_action = action.next_hop_index

        # ---- Advance traffic simulation ---------------------------------
        self._traffic.advance()
        self._refill_packet_queue(difficulty)

        done = self._state.step_count >= self._max_steps

        if done:
            success_threshold = 4.0 * self._max_steps
            success = self._episode_reward >= success_threshold
            self.curriculum.record(
                task_type=self.task_type,
                success=success,
                steps=self._episode_step,
                reward=self._episode_reward,
            )

        avg_lat = (
            self._total_delivery_latency / self._packets_delivered
            if self._packets_delivered
            else 0.0
        )

        return StepResult(
            observation=self._build_observation(),
            reward=float(reward_val),
            done=done,
            info={
                "error": error_tag or None,
                "packets_delivered": self._packets_delivered,
                "packets_dropped": self._packets_dropped,
                "avg_delivery_latency_ms": round(avg_lat, 2),
                "episode_step": self._episode_step,
                "max_steps": self._max_steps,
                "difficulty": round(difficulty, 3),
                "curriculum_tier": self.curriculum.get_tier_name(),
                "graph_nodes": len(self._graph.nodes),
                "graph_links": self._graph.link_count() // 2,
                "active_flows": len(self._traffic.active_flows),
                "episode_reward": round(self._episode_reward, 3),
                "consistency_bonus": consistency_bonus,
                **info_extra,
            },
        )

    @property
    def state(self) -> State:
        return self._state

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        packet: PacketInfo,
        next_hop: str,
        link_lat: float,
        link_util: float,
        total_latency: float,
        delivered: bool,
    ) -> float:
        pmult = _PRIORITY_MULT[min(packet.priority, 2)]

        # 1. Delivery bonus
        if delivered:
            delivery_bonus = 10.0
            optimal_lat = self._graph.diameter * 20.0
            latency_quality = max(0.0, 1.0 - total_latency / (optimal_lat * 3.0))
            delivery_bonus += latency_quality * 5.0   # up to +5 for fast delivery
        else:
            delivery_bonus = -3.0

        # 2. Per-hop latency cost (50 ms = -1.0)
        latency_cost = -(link_lat / 50.0)

        # 3. Congestion cost (quadratic past 70 % utilisation)
        if link_util > 0.70:
            cong_cost = -((link_util - 0.70) / 0.30) ** 2 * 4.0
        else:
            cong_cost = 0.0

        # 4. Progress reward (hop distance to destination)
        agent = self._graph.agent_router
        hops_before = self._hop_dist.get(agent, {}).get(packet.dst, 0)
        hops_after = self._hop_dist.get(next_hop, {}).get(packet.dst, 0)
        progress = hops_before - hops_after   # +1 = closer, 0 = sideways, -1 = farther
        progress_reward = progress * 2.0

        # 5. Load-balance bonus / bad-deviation penalty
        fib_hops = self._graph.fib.get(agent, {}).get(packet.dst, [])
        fib_primary = fib_hops[0] if fib_hops else None
        lb_bonus = 0.0
        if fib_primary and next_hop != fib_primary:
            fib_link = (agent, fib_primary)
            fib_util = self._traffic.get_link_utilization(fib_link)
            if fib_util > 0.75 and link_util < 0.50:
                lb_bonus = 1.5    # smart deviation: avoided congested FIB path
            elif fib_util < 0.50 and link_util > 0.75:
                lb_bonus = -1.0   # bad deviation: chose congested over clear FIB path

        total = delivery_bonus + latency_cost + cong_cost + progress_reward + lb_bonus
        return total * pmult

    # ------------------------------------------------------------------
    # Remainder simulation (FIB-driven auto-routing after agent's hop)
    # ------------------------------------------------------------------

    def _simulate_remainder(
        self,
        from_node: str,
        dst: str,
        ttl: int,
        visited: List[str],
    ) -> _RemainderResult:
        """
        Auto-route the packet from *from_node* to *dst* using the FIB.
        Returns a _RemainderResult describing whether the packet was delivered
        and the total accumulated latency of the remainder path.
        """
        if from_node == dst:
            return _RemainderResult(delivered=True, latency_ms=0.0, hops=0)

        current = from_node
        total_lat = 0.0
        hops = 0
        visited = list(visited)

        while current != dst:
            if ttl <= 0:
                return _RemainderResult(
                    delivered=False, latency_ms=total_lat,
                    hops=hops, drop_reason="TTL_EXHAUSTED",
                )

            fib_options = self._graph.fib.get(current, {}).get(dst, [])
            if not fib_options:
                return _RemainderResult(
                    delivered=False, latency_ms=total_lat,
                    hops=hops, drop_reason="NO_ROUTE",
                )

            # Pick least-loaded option that doesn't create a cycle
            valid_options = [n for n in fib_options if n not in visited]
            if not valid_options:
                return _RemainderResult(
                    delivered=False, latency_ms=total_lat,
                    hops=hops, drop_reason="ALL_OPTIONS_VISITED",
                )

            next_node = min(
                valid_options,
                key=lambda n: self._traffic.get_link_utilization((current, n)),
            )

            link = (current, next_node)
            hop_lat = self._traffic.get_effective_latency(link)
            loss = self._graph.links[link].base_loss

            if random.random() < loss:
                return _RemainderResult(
                    delivered=False, latency_ms=total_lat,
                    hops=hops, drop_reason="STOCHASTIC_LOSS",
                )

            total_lat += hop_lat
            visited.append(current)
            current = next_node
            ttl -= 1
            hops += 1

        return _RemainderResult(delivered=True, latency_ms=total_lat, hops=hops)

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> SmartRouterObservation:
        router = self._graph.agent_router
        neighbors = self._graph.neighbors(router)

        link_states: List[LinkState] = []
        for i, nbr in enumerate(neighbors):
            link = (router, nbr)
            lat = self._traffic.get_effective_latency(link)
            util = self._traffic.get_link_utilization(link)
            link_states.append(
                LinkState(
                    index=i,
                    neighbor=nbr,
                    latency_ms=round(lat, 1),
                    utilization=round(util, 3),
                    queue_depth=int(util * 20),
                    is_congested=util > 0.75,
                )
            )

        fib_entries: List[FIBEntry] = [
            FIBEntry(destination=dst, next_hops=hops)
            for dst, hops in self._graph.fib.get(router, {}).items()
        ]

        # Use next pending packet (or generate one if queue somehow empty)
        if self._pending_packets:
            packet = self._pending_packets[0]
        else:
            packet = self._generate_packet()
            self._pending_packets.append(packet)

        all_utils = self._traffic.get_all_link_utilizations()
        avg_util = (
            sum(all_utils.values()) / len(all_utils) if all_utils else 0.0
        )

        return SmartRouterObservation(
            agent_router=router,
            packet=packet,
            link_states=link_states,
            fib=fib_entries,
            queue_size=len(self._pending_packets),
            active_flow_count=len(self._traffic.active_flows),
            network_utilization=round(avg_util, 3),
        )

    # ------------------------------------------------------------------
    # Packet management
    # ------------------------------------------------------------------

    def _generate_packet(self) -> PacketInfo:
        """Generate a packet with a random src/dst pair (dst ≠ agent router)."""
        possible_dsts = [n for n in self._graph.nodes if n != self._graph.agent_router]
        dst = random.choice(possible_dsts)
        possible_srcs = [n for n in self._graph.nodes if n != dst]
        src = random.choice(possible_srcs)

        return PacketInfo(
            packet_id=str(uuid4())[:8],
            src=src,
            dst=dst,
            priority=random.choices([0, 1, 2], weights=[40, 40, 20])[0],
            ttl=self._graph.diameter * 3,
            hops_taken=0,
            visited=[src],
        )

    def _refill_packet_queue(self, difficulty: float) -> None:
        """
        Keep the queue filled; more packets queued at higher difficulty.
        difficulty 0 → 1 packet; difficulty 1 → 5 packets.
        """
        target = max(1, int(1 + difficulty * 4))
        while len(self._pending_packets) < target:
            self._pending_packets.append(self._generate_packet())

    # ------------------------------------------------------------------
    # BFS hop distances (for progress reward)
    # ------------------------------------------------------------------

    def _bfs_all_distances(self) -> Dict[str, Dict[str, int]]:
        """
        Compute BFS hop counts between every pair of nodes.
        Used in the progress-reward component of _compute_reward.
        """
        dist: Dict[str, Dict[str, int]] = {}
        for start in self._graph.nodes:
            d: Dict[str, int] = {start: 0}
            queue = deque([start])
            while queue:
                node = queue.popleft()
                for nbr in self._graph.neighbors(node):
                    if nbr not in d:
                        d[nbr] = d[node] + 1
                        queue.append(nbr)
            dist[start] = d
        return dist
