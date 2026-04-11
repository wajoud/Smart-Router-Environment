"""
Network topology definitions and FIB computation for the Smart Router environment.

The agent always controls router R2, which sits at high betweenness centrality in
every blueprint topology so that many packets naturally flow through it.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LinkSpec:
    src: str
    dst: str
    base_latency_ms: float   # nominal one-way latency
    capacity_mbps: float     # bandwidth ceiling (for utilisation calculation)
    base_loss: float         # baseline random packet-loss probability [0, 1)


@dataclass
class NetworkGraph:
    nodes: List[str]
    links: Dict[Tuple[str, str], LinkSpec]   # directed; both (A,B) and (B,A) stored
    agent_router: str                        # node the RL agent controls
    diameter: int                            # longest shortest-path hop count
    fib: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    # fib layout: fib[src][dst] = [primary_next_hop, alt_next_hop, ...]

    def neighbors(self, node: str) -> List[str]:
        """Return direct neighbours of *node* in link-index order."""
        return [dst for (src, dst) in self.links if src == node]

    def link_count(self) -> int:
        return len(self.links)


# ---------------------------------------------------------------------------
# Blueprint helpers
# ---------------------------------------------------------------------------

def _bidirectional(
    specs: List[Tuple[str, str, float, float, float]]
) -> Dict[Tuple[str, str], LinkSpec]:
    """
    Build a symmetric link dict from (src, dst, latency, capacity, loss) tuples.
    Each entry creates both (src→dst) and (dst→src) with the same parameters.
    """
    links: Dict[Tuple[str, str], LinkSpec] = {}
    for src, dst, lat, cap, loss in specs:
        links[(src, dst)] = LinkSpec(src, dst, lat, cap, loss)
        links[(dst, src)] = LinkSpec(dst, src, lat, cap, loss)
    return links


# ---------------------------------------------------------------------------
# Blueprint topologies
# Nodes are R0…R13.  R2 is always the agent router (central position).
# ---------------------------------------------------------------------------

def _build_warmup() -> NetworkGraph:
    """
    5 nodes, 6 bidirectional links.
    Ring R0-R1-R2-R3-R4-R0 plus a shortcut R1-R3.
    R2 sits in the middle of the ring — all traffic crosses it.
    """
    specs = [
        ("R0", "R1", 10.0, 100.0, 0.005),
        ("R1", "R2", 12.0, 100.0, 0.005),
        ("R2", "R3", 15.0, 100.0, 0.005),
        ("R3", "R4", 10.0, 100.0, 0.005),
        ("R4", "R0", 20.0, 100.0, 0.008),
        ("R1", "R3", 25.0,  80.0, 0.010),   # shortcut
    ]
    return NetworkGraph(
        nodes=["R0", "R1", "R2", "R3", "R4"],
        links=_bidirectional(specs),
        agent_router="R2",
        diameter=3,
    )


def _build_beginner() -> NetworkGraph:
    """
    7 nodes, 10 bidirectional links.
    Two rings sharing R2 as the bridge node.
    """
    specs = [
        # Left ring: R0-R1-R2-R5-R0
        ("R0", "R1", 10.0, 100.0, 0.005),
        ("R1", "R2", 12.0, 100.0, 0.005),
        ("R2", "R5", 18.0, 100.0, 0.006),
        ("R5", "R0", 22.0,  80.0, 0.008),
        # Right ring: R2-R3-R4-R6-R2
        ("R2", "R3", 15.0, 100.0, 0.005),
        ("R3", "R4", 10.0, 100.0, 0.005),
        ("R4", "R6", 20.0,  80.0, 0.007),
        ("R6", "R2", 28.0,  80.0, 0.010),
        # Cross-links for redundancy
        ("R1", "R3", 30.0,  60.0, 0.012),
        ("R0", "R4", 35.0,  60.0, 0.015),
    ]
    return NetworkGraph(
        nodes=["R0", "R1", "R2", "R3", "R4", "R5", "R6"],
        links=_bidirectional(specs),
        agent_router="R2",
        diameter=4,
    )


def _build_intermediate() -> NetworkGraph:
    """
    9 nodes, 14 bidirectional links.
    Partial mesh with R2 at centre.
    """
    specs = [
        # Core star from R2
        ("R2", "R1", 10.0, 150.0, 0.004),
        ("R2", "R3", 12.0, 150.0, 0.004),
        ("R2", "R5", 15.0, 120.0, 0.005),
        ("R2", "R7", 20.0, 100.0, 0.006),
        # Outer ring
        ("R0", "R1", 18.0, 100.0, 0.006),
        ("R1", "R4", 22.0,  80.0, 0.008),
        ("R4", "R3", 15.0,  80.0, 0.006),
        ("R3", "R6", 25.0,  80.0, 0.009),
        ("R6", "R5", 20.0,  80.0, 0.007),
        ("R5", "R8", 18.0, 100.0, 0.006),
        ("R8", "R7", 22.0,  80.0, 0.008),
        ("R7", "R0", 30.0,  60.0, 0.012),
        # Shortcuts
        ("R0", "R4", 35.0,  60.0, 0.015),
        ("R6", "R8", 28.0,  60.0, 0.011),
    ]
    return NetworkGraph(
        nodes=["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"],
        links=_bidirectional(specs),
        agent_router="R2",
        diameter=5,
    )


def _build_advanced() -> NetworkGraph:
    """
    11 nodes, 20 bidirectional links.
    Dense mesh; R2 bridges two sub-clusters.
    """
    specs = [
        # Core links from R2
        ("R2", "R1", 8.0,  200.0, 0.003),
        ("R2", "R3", 10.0, 200.0, 0.003),
        ("R2", "R5", 12.0, 150.0, 0.004),
        ("R2", "R8", 18.0, 120.0, 0.005),
        # Left cluster R0-R1-R4-R6
        ("R0", "R1", 15.0, 100.0, 0.005),
        ("R1", "R4", 20.0,  80.0, 0.007),
        ("R4", "R6", 18.0,  80.0, 0.006),
        ("R6", "R0", 25.0,  80.0, 0.009),
        ("R0", "R4", 28.0,  60.0, 0.011),
        # Right cluster R3-R5-R7-R9-R10
        ("R3", "R7", 15.0, 100.0, 0.005),
        ("R5", "R9", 20.0,  80.0, 0.007),
        ("R7", "R9", 18.0,  80.0, 0.006),
        ("R9", "R10", 12.0, 100.0, 0.005),
        ("R10", "R3", 22.0,  80.0, 0.008),
        ("R7", "R10", 30.0,  60.0, 0.012),
        # Inter-cluster bridges
        ("R6", "R8", 35.0,  60.0, 0.014),
        ("R4", "R5", 30.0,  60.0, 0.012),
        ("R1", "R3", 25.0,  80.0, 0.009),
        ("R8", "R9", 20.0,  80.0, 0.007),
        ("R6", "R7", 40.0,  50.0, 0.016),
    ]
    return NetworkGraph(
        nodes=["R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"],
        links=_bidirectional(specs),
        agent_router="R2",
        diameter=6,
    )


def _build_expert() -> NetworkGraph:
    """
    14 nodes, 28 bidirectional links.
    Fully irregular mesh; R2 is the high-betweenness backbone node.
    """
    specs = [
        # Backbone links from R2
        ("R2", "R1",  8.0, 250.0, 0.002),
        ("R2", "R3", 10.0, 250.0, 0.002),
        ("R2", "R5", 12.0, 200.0, 0.003),
        ("R2", "R8", 15.0, 150.0, 0.004),
        ("R2", "R11", 20.0, 120.0, 0.005),
        # Left cluster
        ("R0", "R1",  12.0, 100.0, 0.005),
        ("R1", "R4",  18.0,  80.0, 0.007),
        ("R4", "R6",  15.0,  80.0, 0.006),
        ("R6", "R0",  22.0,  80.0, 0.008),
        ("R0", "R4",  25.0,  60.0, 0.010),
        ("R6", "R13", 30.0,  60.0, 0.012),
        # Middle cluster
        ("R3", "R7",  14.0, 100.0, 0.005),
        ("R5", "R9",  18.0,  80.0, 0.007),
        ("R7", "R9",  16.0,  80.0, 0.006),
        ("R9", "R12", 12.0, 100.0, 0.004),
        ("R12", "R3", 20.0,  80.0, 0.007),
        ("R7", "R12", 28.0,  60.0, 0.011),
        # Right cluster
        ("R8",  "R10", 15.0, 100.0, 0.005),
        ("R10", "R11", 20.0,  80.0, 0.007),
        ("R11", "R13", 18.0,  80.0, 0.006),
        ("R13", "R8",  25.0,  60.0, 0.010),
        ("R10", "R13", 30.0,  60.0, 0.012),
        # Inter-cluster cross-links
        ("R4",  "R5",  28.0,  60.0, 0.011),
        ("R1",  "R3",  22.0,  80.0, 0.008),
        ("R6",  "R7",  35.0,  50.0, 0.014),
        ("R8",  "R9",  18.0,  80.0, 0.006),
        ("R12", "R11", 25.0,  60.0, 0.010),
        ("R13", "R9",  32.0,  50.0, 0.013),
    ]
    return NetworkGraph(
        nodes=["R0","R1","R2","R3","R4","R5","R6","R7",
               "R8","R9","R10","R11","R12","R13"],
        links=_bidirectional(specs),
        agent_router="R2",
        diameter=7,
    )


# ---------------------------------------------------------------------------
# FIB computation (Dijkstra)
# ---------------------------------------------------------------------------

def _dijkstra(
    graph: NetworkGraph, source: str
) -> Tuple[Dict[str, float], Dict[str, Optional[str]]]:
    """
    Compute shortest paths from *source* using base_latency_ms as edge weight.
    Returns (dist, prev) where prev[node] is the node immediately before *node*
    on the shortest path from *source*.
    """
    dist: Dict[str, float] = {n: float("inf") for n in graph.nodes}
    prev: Dict[str, Optional[str]] = {n: None for n in graph.nodes}
    dist[source] = 0.0
    heap = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue
        for v in graph.neighbors(u):
            alt = d + graph.links[(u, v)].base_latency_ms
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(heap, (alt, v))

    return dist, prev


def _reconstruct_path(prev: Dict[str, Optional[str]], source: str, dst: str) -> List[str]:
    """Trace back the predecessor map to build the full path source→dst."""
    path: List[str] = []
    node: Optional[str] = dst
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    if not path or path[0] != source:
        return []
    return path


def compute_fib(graph: NetworkGraph) -> Dict[str, Dict[str, List[str]]]:
    """
    For every (src, dst) pair, compute the top-2 next-hops from *src* toward *dst*.

    The primary next-hop is from the shortest-path tree.
    The alternate next-hop is found by temporarily removing the primary edge and
    re-running Dijkstra — a simple loop-free alternate (LFA) approximation.
    """
    fib: Dict[str, Dict[str, List[str]]] = {}

    for source in graph.nodes:
        fib[source] = {}
        dist, prev = _dijkstra(graph, source)

        for dst in graph.nodes:
            if dst == source:
                continue

            path = _reconstruct_path(prev, source, dst)
            if len(path) < 2:
                continue   # unreachable

            primary_next_hop = path[1]
            next_hops = [primary_next_hop]

            # Find an alternate next-hop (LFA): a direct neighbour of source that
            # can reach dst via a path shorter than going through source itself.
            # dist_via_neighbour[dst] < dist[dst] means loop-free.
            for nbr in graph.neighbors(source):
                if nbr == primary_next_hop:
                    continue
                # Dijkstra from neighbour's perspective
                nbr_dist, _ = _dijkstra(graph, nbr)
                link_cost = graph.links[(source, nbr)].base_latency_ms
                if nbr_dist.get(dst, float("inf")) + link_cost < dist[dst] * 1.5:
                    next_hops.append(nbr)
                    break   # one alternate is enough

            fib[source][dst] = next_hops

    return fib


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

_BLUEPRINTS: Dict[str, NetworkGraph] = {
    "warmup":       _build_warmup(),
    "beginner":     _build_beginner(),
    "intermediate": _build_intermediate(),
    "advanced":     _build_advanced(),
    "expert":       _build_expert(),
}


def _difficulty_to_tier(difficulty: float) -> str:
    if difficulty < 0.25:
        return "warmup"
    elif difficulty < 0.40:
        return "beginner"
    elif difficulty < 0.60:
        return "intermediate"
    elif difficulty < 0.80:
        return "advanced"
    else:
        return "expert"


def generate_topology(difficulty: float) -> NetworkGraph:
    """
    Return a *new* NetworkGraph instance scaled to the given difficulty.
    The FIB is pre-computed before returning.
    """
    tier = _difficulty_to_tier(difficulty)
    blueprint = _BLUEPRINTS[tier]

    # Return a fresh copy so the caller can mutate fib without side-effects
    graph = NetworkGraph(
        nodes=list(blueprint.nodes),
        links=dict(blueprint.links),
        agent_router=blueprint.agent_router,
        diameter=blueprint.diameter,
    )
    graph.fib = compute_fib(graph)
    return graph
