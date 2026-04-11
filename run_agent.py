"""
Example reactive agent for the Smart Router Environment.

Strategy:
  1. Check the FIB for the suggested next-hop toward the packet's destination.
  2. If the FIB-suggested link is congested (util > 0.75), look for an
     uncongested alternative neighbour that makes progress toward the destination.
  3. If all links are congested, pick the least-loaded one.
  4. Never choose a neighbour already in packet.visited (cycle avoidance).
  5. Drop (-1) if TTL is critically low and all neighbours would loop.
"""

import asyncio
import httpx


async def run_routing_agent():
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        print("Agent Connected. Starting Network Optimization...")

        # Reset to get the starting observation
        response = await client.post("/reset")
        data = response.json()
        current_obs = _extract_obs(data)

        step = 0
        while True:
            step += 1

            action_idx = _decide(current_obs)

            response = await client.post(
                "/step", json={"action": {"next_hop_index": action_idx}}
            )
            data = response.json()

            current_obs = _extract_obs(data)
            reward = data.get("reward", 0.0)
            done = data.get("done", False)
            info = data.get("info", {})

            # ---- logging ------------------------------------------------
            pkt = current_obs.get("packet", {})
            links = current_obs.get("link_states", [])
            chosen_link = next((l for l in links if l.get("index") == action_idx), None)

            if action_idx == -1:
                action_str = "DROP"
            elif chosen_link:
                nbr = chosen_link["neighbor"]
                lat = chosen_link["latency_ms"]
                util = chosen_link["utilization"]
                action_str = f"idx={action_idx}({nbr}) {lat:.1f}ms util={util:.2f}"
            else:
                action_str = f"idx={action_idx}(invalid)"

            delivered = info.get("delivered", "?")
            err = info.get("error") or ""
            flows = info.get("active_flows", "?")
            net_util = current_obs.get("network_utilization", 0.0)

            print(
                f"Step {step:4d} | {action_str:40s} | "
                f"reward={reward:+7.2f} | "
                f"pkt={pkt.get('packet_id','?')} {pkt.get('src','?')}->{pkt.get('dst','?')} "
                f"pri={pkt.get('priority','?')} ttl={pkt.get('ttl','?')} | "
                f"delivered={delivered} | flows={flows} net={net_util:.2f} | "
                f"{'ERR:'+err if err else ''}"
            )

            if done:
                print(
                    f"\nEpisode done after {step} steps | "
                    f"delivered={info.get('packets_delivered','?')} "
                    f"dropped={info.get('packets_dropped','?')} | "
                    f"avg_lat={info.get('avg_delivery_latency_ms','?')}ms | "
                    f"tier={info.get('curriculum_tier','?')} "
                    f"difficulty={info.get('difficulty','?')}"
                )
                break


def _extract_obs(data: dict) -> dict:
    """Unwrap the observation regardless of nesting."""
    raw = data.get("observation", data)
    if isinstance(raw, dict):
        return raw.get("observation", raw)
    return raw


def _decide(obs: dict) -> int:
    """
    FIB-following greedy decision with congestion avoidance.
    Returns next_hop_index (or -1 to drop).
    """
    packet = obs.get("packet", {})
    dst = packet.get("dst", "")
    visited = set(packet.get("visited", []))
    ttl = packet.get("ttl", 0)
    links: list = obs.get("link_states", [])
    fib: list = obs.get("fib", [])

    if not links:
        return -1

    # Build a map: neighbour → link state
    link_by_nbr = {l["neighbor"]: l for l in links}

    # Exclude already-visited neighbours and those that would loop
    valid_links = [l for l in links if l["neighbor"] not in visited]

    if not valid_links:
        return -1   # all neighbours visited → drop

    if ttl <= 2:
        # About to expire — drop is cheaper than the TTL penalty
        return -1

    # FIB suggestion for this destination
    fib_suggestion = _fib_lookup(fib, dst)

    if fib_suggestion and fib_suggestion not in visited:
        fib_link = link_by_nbr.get(fib_suggestion)
        if fib_link and not fib_link["is_congested"]:
            return fib_link["index"]   # FIB primary is clear — use it

    # FIB is congested or absent — find best valid alternative
    # Prefer: not congested → lowest (utilization × latency) cost
    uncongested = [l for l in valid_links if not l["is_congested"]]
    pool = uncongested if uncongested else valid_links

    best = min(pool, key=lambda l: l["utilization"] * l["latency_ms"])
    return best["index"]


def _fib_lookup(fib: list, dst: str) -> str | None:
    """Return the primary next-hop for *dst* from the FIB, or None."""
    for entry in fib:
        if entry.get("destination") == dst:
            hops = entry.get("next_hops", [])
            return hops[0] if hops else None
    return None


if __name__ == "__main__":
    asyncio.run(run_routing_agent())
