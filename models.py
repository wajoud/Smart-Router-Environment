# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Smart Router Environment.

The agent acts as a specific router (R2) in a multi-hop network graph.
Each step it observes one packet that needs to be forwarded and decides
which neighbouring router to send it to.
"""

from typing import Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models (plain BaseModel — not Action/Observation)
# ---------------------------------------------------------------------------

class PacketInfo(BaseModel):
    """A packet currently awaiting forwarding at the agent router."""

    packet_id: str
    src: str                 # originating router
    dst: str                 # destination router
    priority: int            # 0 = low, 1 = medium, 2 = high
    ttl: int                 # time-to-live: drops when it reaches 0
    hops_taken: int          # number of hops already completed
    visited: List[str]       # routers already traversed (cycle detection)


class LinkState(BaseModel):
    """Current state of one outgoing link from the agent router."""

    index: int               # action index — pass this number as next_hop_index
    neighbor: str            # router name this link connects to
    latency_ms: float        # current effective latency (base + queuing delay)
    utilization: float       # 0.0–1.0 fraction of capacity used by background traffic
    queue_depth: int         # approximate number of packets queued on this link
    is_congested: bool       # True when utilization > 0.75


class FIBEntry(BaseModel):
    """One row of the agent router's Forwarding Information Base."""

    destination: str
    next_hops: List[str]     # [primary_next_hop, alt_next_hop] computed by Dijkstra


# ---------------------------------------------------------------------------
# Top-level Action and Observation
# ---------------------------------------------------------------------------

class SmartRouterAction(Action):
    """
    Forwarding decision for the current packet.

    next_hop_index must be:
      -1          → intentionally drop the packet (mild penalty for high-priority)
      0 … N-1     → index into link_states (forward to that neighbour)

    Choosing an index that is out of range, or choosing a neighbour already in
    packet.visited, results in a large negative reward and the packet is dropped.
    """

    next_hop_index: int = Field(..., ge=-1)


class SmartRouterObservation(Observation):
    """Full observation returned after each reset() or step()."""

    agent_router: str                    # the router the agent IS (always "R2")
    packet: PacketInfo                   # packet currently awaiting a forwarding decision
    link_states: List[LinkState]         # one entry per neighbour, ordered by index
    fib: List[FIBEntry]                  # forwarding table (Dijkstra-computed suggestions)
    queue_size: int                      # number of packets pending at agent router
    active_flow_count: int               # background flows currently active
    network_utilization: float           # average utilization across all network links
