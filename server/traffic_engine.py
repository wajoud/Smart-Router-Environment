"""
Background traffic simulation for the Smart Router environment.

TrafficEngine maintains a set of active flows and burst events that load
network links independently of the RL agent's forwarding decisions.
This creates realistic, dynamic congestion patterns the agent must react to.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from .topology import NetworkGraph, LinkSpec
except ImportError:
    from server.topology import NetworkGraph, LinkSpec


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Flow:
    src: str
    dst: str
    load_mbps: float
    path: List[str]           # ordered list of nodes (includes src and dst)
    remaining_steps: int      # steps until this flow expires
    is_burst: bool            # True → high-load spike event


# ---------------------------------------------------------------------------
# Traffic engine
# ---------------------------------------------------------------------------

class TrafficEngine:
    """
    Simulates background network traffic on top of the agent's decisions.

    Each step:
    1. Existing flows are aged; expired ones are removed.
    2. New regular flows are spawned to maintain the target count.
    3. With some probability, a burst event is injected.
    4. Per-link loads are recomputed as the sum of loads from all flows
       whose paths traverse that link.

    The engine exposes effective latency per link, which includes a queuing
    delay that grows as utilisation approaches 100 % (M/D/1 approximation).
    """

    def __init__(self, graph: NetworkGraph, difficulty: float) -> None:
        self.graph = graph
        self.difficulty = difficulty
        self.active_flows: List[Flow] = []

        # Initialise all link loads to 0
        self._link_load_mbps: Dict[Tuple[str, str], float] = {
            k: 0.0 for k in graph.links
        }

        # Scaling parameters from difficulty
        self._max_flows = self._flows_for_difficulty(difficulty)
        self._burst_prob = 0.05 + difficulty * 0.15   # 5 % → 20 % per step
        self._base_flow_load = (10.0, 80.0)            # Mbps range for regular flows
        self._burst_load = (300.0, 600.0)              # Mbps range for burst spikes

        # Seed initial flows so the network is already loaded at episode start
        for _ in range(min(self._max_flows, 3)):
            self._spawn_flow(is_burst=False)
        self._recompute_link_loads()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def advance(self) -> None:
        """Advance the traffic simulation by one step."""
        self._expire_flows()
        self._spawn_new_flows()
        self._maybe_trigger_burst()
        self._recompute_link_loads()

    def get_link_utilization(self, link: Tuple[str, str]) -> float:
        """
        Return the fraction of link capacity consumed by background traffic [0, 1].
        Does not account for the agent's own packet (that is marginal).
        """
        spec = self.graph.links.get(link)
        if spec is None:
            return 0.0
        load = self._link_load_mbps.get(link, 0.0)
        return min(1.0, load / spec.capacity_mbps)

    def get_effective_latency(self, link: Tuple[str, str]) -> float:
        """
        Compute current one-way latency for *link* in milliseconds.

        Model:
          - Below 70 % utilisation: only base latency + small jitter.
          - Above 70 %: queuing delay grows rapidly (M/D/1 approximation):
              extra_delay = base_lat * ρ² / (2 * (1 - ρ))   where ρ = utilisation
          - Plus Gaussian jitter scaled to difficulty.
        """
        spec = self.graph.links.get(link)
        if spec is None:
            return 999.0

        util = self.get_link_utilization(link)
        base = spec.base_latency_ms

        # Queuing delay (M/D/1)
        if util >= 0.99:
            queuing = base * 10.0   # saturated link — very high delay
        elif util > 0.70:
            rho = util
            queuing = base * (rho ** 2) / (2.0 * (1.0 - rho))
        else:
            queuing = 0.0

        # Jitter
        jitter_std = base * 0.04 * (1.0 + self.difficulty)
        jitter = random.gauss(0.0, jitter_std)

        return max(0.1, base + queuing + jitter)

    def get_all_link_utilizations(self) -> Dict[Tuple[str, str], float]:
        """Return {link: utilization} for every link in the graph."""
        return {lk: self.get_link_utilization(lk) for lk in self.graph.links}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _flows_for_difficulty(difficulty: float) -> int:
        """Scale max concurrent background flows with difficulty."""
        # warmup=2, beginner=5, intermediate=10, advanced=18, expert=28
        return max(2, int(2 + difficulty * 26))

    def _shortest_path(self, src: str, dst: str) -> List[str]:
        """Return shortest-hop path src → dst using the precomputed FIB."""
        if src == dst:
            return [src]
        path = [src]
        current = src
        visited = {src}
        while current != dst:
            fib_hops = self.graph.fib.get(current, {}).get(dst, [])
            if not fib_hops:
                return []
            next_node = fib_hops[0]
            if next_node in visited:
                return []   # FIB loop — give up
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        return path

    def _spawn_flow(self, is_burst: bool) -> None:
        """Create and register a single new flow."""
        nodes = self.graph.nodes
        if len(nodes) < 2:
            return
        src, dst = random.sample(nodes, 2)
        path = self._shortest_path(src, dst)
        if len(path) < 2:
            return

        if is_burst:
            load = random.uniform(*self._burst_load)
            duration = random.randint(3, 10)
        else:
            load = random.uniform(*self._base_flow_load)
            duration = random.randint(5, 25)

        self.active_flows.append(Flow(src, dst, load, path, duration, is_burst))

    def _spawn_new_flows(self) -> None:
        """Top up regular flows to the current maximum."""
        regular = [f for f in self.active_flows if not f.is_burst]
        while len(regular) < self._max_flows:
            self._spawn_flow(is_burst=False)
            regular.append(self.active_flows[-1])   # keep count in sync

    def _maybe_trigger_burst(self) -> None:
        """Randomly inject a burst event on a random path."""
        if random.random() < self._burst_prob:
            self._spawn_flow(is_burst=True)

    def _expire_flows(self) -> None:
        """Decrement counters and remove expired flows."""
        surviving: List[Flow] = []
        for flow in self.active_flows:
            flow.remaining_steps -= 1
            if flow.remaining_steps > 0:
                surviving.append(flow)
        self.active_flows = surviving

    def _recompute_link_loads(self) -> None:
        """Sum the load of every active flow across each link it traverses."""
        loads: Dict[Tuple[str, str], float] = {k: 0.0 for k in self.graph.links}
        for flow in self.active_flows:
            for i in range(len(flow.path) - 1):
                link = (flow.path[i], flow.path[i + 1])
                if link in loads:
                    loads[link] += flow.load_mbps
        self._link_load_mbps = loads
