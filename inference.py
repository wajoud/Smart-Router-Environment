"""
Inference Script for Smart Router Environment
==============================================

PRIMARY EVALUATION FILE — runs all five task scenarios and validates the entire
multi-hop network simulation:

  routing-basic       warmup topology   (5 nodes), low background traffic
  routing-congested   intermediate      (9 nodes), heavy background flows
  routing-burst       intermediate      (9 nodes), frequent burst events
  routing-multihop    advanced          (11 nodes), many hops required
  routing-expert      expert            (14 nodes), max flows + frequent bursts

MANDATORY REQUIREMENTS:
  API_BASE_URL   LLM endpoint            (default: https://router.huggingface.co/v1)
  MODEL_NAME     Model identifier        (default: Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       API key

  The environment runs entirely in-process — no server or Docker required.
  This file must be named inference.py and placed in the project root.
  Use the OpenAI client for LLM calls.

STDOUT FORMAT
  [START] task=<name> env=smart_router model=<model> nodes=<n> links=<l> agent=<r>
  [STEP]  step=<n> action=<idx>(<nbr>) reward=<r> done=<bool> ...details...
  [END]   success=<bool> steps=<n> score=<s> rewards=<r1,...>
          delivered=<n> dropped=<n> delivery_rate=<f> avg_latency=<f>ms

  Example:
    [START] task=routing-expert env=smart_router model=Qwen2.5-72B nodes=14 links=28 agent=R2
    [STEP] step=1 action=1(R3) reward=+12.40 done=false packet=a3f1 src=R0 dst=R9 \\
           pri=high ttl=21 fib_suggested=R3 delivered=true total_lat=67.2ms \\
           link_lat=10.3ms util=0.28 active_flows=22 net_util=0.61 error=null
    [STEP] step=2 action=0(R1) reward=-16.00 done=false packet=b7c2 src=R4 dst=R8 \\
           pri=high ttl=3 error=CYCLE:R1_already_in_visited
    [END] success=true steps=50 score=0.712 rewards=12.40,-16.00,...
          delivered=38 dropped=12 delivery_rate=0.76 avg_latency=71.3ms
"""

import os
import textwrap
import time
from typing import List, Optional

from openai import OpenAI
from smart_router import SmartRouterAction, SmartRouterObservation

from server.smart_router_environment import SmartRouterEnvironment

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")

BENCHMARK = os.getenv("SMART_ROUTER_BENCHMARK", "smart_router")
MAX_STEPS = 50
TEMPERATURE = 0.3
MAX_TOKENS = 20  # agent only needs to output a single integer

# Five task types that together exercise every feature of the simulation
TASK_TYPES = [
    "routing-basic",
    "routing-congested",
    "routing-burst",
    "routing-multihop",
    "routing-expert",
]

# Score calibration: optimal agent delivers ~12 reward/step on high-priority pkts
_MAX_REWARD_PER_STEP = 12.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SUCCESS_SCORE_THRESHOLD = 0.35  # normalised [0, 1]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are router R2 in a multi-hop IP network. Your job is to forward each
    arriving packet to the best next-hop router.

    ── HOW TO READ YOUR OBSERVATION ──────────────────────────────────────────
    Each step you see:
      • PACKET  — the packet to forward: source, destination, priority,
                  TTL (time-to-live), and visited[] (routers already traversed)
      • LINKS   — your outgoing links, each with:
                    index      : the integer to output as your action
                    neighbor   : name of the router this link connects to
                    latency_ms : current one-way delay (base + queuing)
                    utilization: fraction of capacity used (0.0 = idle, 1.0 = full)
                    is_congested: True if utilization > 0.75
      • FIB     — your forwarding table (Dijkstra shortest-path suggestions)
      • Network summary: background flows, average network utilization

    ── RULES ─────────────────────────────────────────────────────────────────
    1. NEVER pick a neighbor that is already in packet.visited  → CYCLE penalty
    2. If TTL ≤ 2, output -1 (drop) — the cost is lower than wasting the hop
    3. Follow the FIB suggestion unless that link is_congested
    4. When the FIB link is congested, pick the least-loaded valid alternative
    5. An index out of range is invalid → large penalty

    ── REWARD SIGNALS ────────────────────────────────────────────────────────
    Delivered packet (fast)  +10 to +15  × priority_mult
    Invalid index            -10         × priority_mult
    Cycle detected           -8          × priority_mult
    TTL expired              -5          × priority_mult
    Intentional drop (-1)    -1          × priority_mult
    Progress toward dst      +2/hop      × priority_mult
    Congestion (util>70 %)   up to -4    × priority_mult
    Load-balance bonus       +1.5        when you smartly avoid congested FIB
    priority_mult: low=0.5, medium=1.0, high=2.0

    ── OUTPUT FORMAT ─────────────────────────────────────────────────────────
    Reply with ONE integer only — the link index (e.g. 0, 1, 2 …) or -1 to drop.
    No punctuation, no explanation, no quotes.  Just the number.
    """
).strip()


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_start(task: str, model: str, nodes: int, links: int, agent: str) -> None:
    print(
        f"[START] task={task} env={BENCHMARK} model={model} "
        f"nodes={nodes} links={links} agent={agent}",
        flush=True,
    )


def log_step(
    step: int,
    action_idx: int,
    neighbor: str,
    reward: float,
    done: bool,
    packet_id: str,
    src: str,
    dst: str,
    priority: str,
    ttl: int,
    fib_suggested: str,
    delivered: Optional[bool],
    total_lat: Optional[float],
    link_lat: Optional[float],
    link_util: Optional[float],
    active_flows: int,
    net_util: float,
    error: Optional[str],
) -> None:
    action_str = f"{action_idx}({neighbor})" if neighbor else str(action_idx)
    delivered_str = str(delivered).lower() if delivered is not None else "n/a"
    lat_str = f"{total_lat:.1f}ms" if total_lat is not None else "n/a"
    link_lat_str = f"{link_lat:.1f}ms" if link_lat is not None else "n/a"
    util_str = f"{link_util:.2f}" if link_util is not None else "n/a"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:+.2f} done={str(done).lower()} "
        f"packet={packet_id} src={src} dst={dst} pri={priority} ttl={ttl} "
        f"fib={fib_suggested} delivered={delivered_str} total_lat={lat_str} "
        f"link_lat={link_lat_str} util={util_str} "
        f"active_flows={active_flows} net_util={net_util:.2f} error={error_str}",
        flush=True,
    )


def log_end(
    success: bool,
    steps: int,
    score: float,
    rewards: List[float],
    delivered: int,
    dropped: int,
    avg_latency: float,
) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    total = delivered + dropped
    rate = delivered / total if total else 0.0
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} "
        f"rewards={rewards_str} "
        f"delivered={delivered} dropped={dropped} "
        f"delivery_rate={rate:.3f} avg_latency={avg_latency:.1f}ms",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_user_prompt(
    step: int,
    obs: SmartRouterObservation,
    last_action: Optional[int],
    last_reward: float,
    history: List[str],
) -> str:
    packet = obs.packet
    priority_label = ["LOW", "MEDIUM", "HIGH"][min(packet.priority, 2)]
    visited_str = ", ".join(packet.visited) if packet.visited else "none"

    # Link table
    link_rows = []
    for ls in obs.link_states:
        cong = " [CONGESTED]" if ls.is_congested else ""
        link_rows.append(
            f"  [{ls.index}] {ls.neighbor:6s} | {ls.latency_ms:6.1f}ms "
            f"| util={ls.utilization:.2f} | qdepth={ls.queue_depth}{cong}"
        )
    links_block = "\n".join(link_rows) if link_rows else "  (no links)"

    # FIB suggestion for this packet's destination
    fib_suggestion = _fib_lookup(obs.fib, packet.dst)
    fib_str = (
        (
            f"primary={fib_suggestion.next_hops[0] if fib_suggestion and fib_suggestion.next_hops else '?'}"
            f", alt={fib_suggestion.next_hops[1] if fib_suggestion and len(fib_suggestion.next_hops) > 1 else 'none'}"
        )
        if fib_suggestion
        else "no FIB entry"
    )

    history_block = "\n".join(history[-4:]) if history else "none"

    return textwrap.dedent(
        f"""
        Step {step} | Router: {obs.agent_router} | Queue: {obs.queue_size} pending
        Background flows: {obs.active_flow_count} | Network utilization: {obs.network_utilization:.0%}

        PACKET: {packet.packet_id}
          src={packet.src}  dst={packet.dst}  priority={priority_label}  TTL={packet.ttl}
          visited=[{visited_str}]

        AVAILABLE LINKS (pick index, or -1 to drop):
        {links_block}

        FIB for {packet.dst}: {fib_str}

        Last action: {last_action if last_action is not None else "none"}
        Last reward: {last_reward:+.2f}

        Recent history:
        {history_block}

        Select next_hop_index:
        """
    ).strip()


def _fib_lookup(fib, dst: str):
    """Return the FIBEntry for *dst*, or None."""
    for entry in fib:
        if entry.destination == dst:
            return entry
    return None


# ---------------------------------------------------------------------------
# Fallback decision (no LLM / parse failure)
# ---------------------------------------------------------------------------


def _fallback_action(obs: SmartRouterObservation) -> int:
    """
    Greedy rule-based decision:
    1. Follow FIB primary if uncongested and not in visited.
    2. Else pick least-loaded uncongested valid neighbour.
    3. Else pick least-loaded valid neighbour.
    4. Else drop (-1).
    """
    packet = obs.packet
    visited = set(packet.visited)
    links = obs.link_states

    if packet.ttl <= 2:
        return -1

    valid = [l for l in links if l.neighbor not in visited]
    if not valid:
        return -1

    fib_entry = _fib_lookup(obs.fib, packet.dst)
    primary = fib_entry.next_hops[0] if fib_entry and fib_entry.next_hops else None

    if primary and primary not in visited:
        primary_link = next((l for l in valid if l.neighbor == primary), None)
        if primary_link and not primary_link.is_congested:
            return primary_link.index

    # Find best alternative
    uncongested = [l for l in valid if not l.is_congested]
    pool = uncongested if uncongested else valid
    best = min(pool, key=lambda l: l.utilization * l.latency_ms)
    return best.index


# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------


def get_model_action(
    client: OpenAI,
    step: int,
    obs: SmartRouterObservation,
    last_action: Optional[int],
    last_reward: float,
    history: List[str],
) -> int:
    user_prompt = build_user_prompt(step, obs, last_action, last_reward, history)

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()

        # Parse: accept -1 or any non-negative integer
        import re

        match = re.search(r"-?\d+", text)
        if match:
            idx = int(match.group())
            if idx == -1 or 0 <= idx < len(obs.link_states):
                return idx

        return _fallback_action(obs)

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return _fallback_action(obs)


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


def run_task(task_name: str, client: Optional[OpenAI]) -> None:
    env = SmartRouterEnvironment(task_type=task_name)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    total_delivered = 0
    total_dropped = 0
    last_avg_latency = 0.0

    try:
        obs = env.reset()

        # Log start with topology metadata
        graph = env._graph
        log_start(
            task=task_name,
            model=MODEL_NAME,
            nodes=len(graph.nodes),
            links=graph.link_count() // 2,
            agent=graph.agent_router,
        )

        last_action: Optional[int] = None
        last_reward = 0.0
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from LLM (or fallback if no client)
            if client is not None:
                action_idx = get_model_action(
                    client, step, obs, last_action, last_reward, history
                )
            else:
                action_idx = _fallback_action(obs)

            # Execute
            result = env.step(SmartRouterAction(next_hop_index=action_idx))
            info = result.info or {}
            new_obs: SmartRouterObservation = result.observation
            reward = result.reward or 0.0
            done = result.done

            rewards.append(reward)
            steps_taken = step
            last_action = action_idx
            last_reward = reward

            # Resolve logged fields
            chosen_link = next(
                (l for l in obs.link_states if l.index == action_idx), None
            )
            neighbor_name = (
                chosen_link.neighbor
                if chosen_link
                else ("DROP" if action_idx == -1 else "?")
            )
            priority_label = ["low", "medium", "high"][min(obs.packet.priority, 2)]

            log_step(
                step=step,
                action_idx=action_idx,
                neighbor=neighbor_name,
                reward=reward,
                done=done,
                packet_id=obs.packet.packet_id,
                src=obs.packet.src,
                dst=obs.packet.dst,
                priority=priority_label,
                ttl=obs.packet.ttl,
                fib_suggested=info.get("fib_suggested", "?"),
                delivered=info.get("delivered"),
                total_lat=info.get("total_latency_ms"),
                link_lat=info.get("link_latency_ms"),
                link_util=info.get("link_utilization"),
                active_flows=info.get("active_flows", 0),
                net_util=new_obs.network_utilization,
                error=info.get("error"),
            )

            # Build history entry for next prompt
            history.append(
                f"Step {step}: idx={action_idx}({neighbor_name}) "
                f"pkt={obs.packet.packet_id} {obs.packet.src}->{obs.packet.dst} "
                f"delivered={info.get('delivered', '?')} "
                f"reward={reward:+.2f}"
            )

            total_delivered = info.get("packets_delivered", 0)
            total_dropped = info.get("packets_dropped", 0)
            last_avg_latency = info.get("avg_delivery_latency_ms", 0.0)

            obs = new_obs

            if done:
                break

        # Compute score
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.001
        score = min(max(score, 0.001), 0.999)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(
            success=success,
            steps=steps_taken,
            score=score,
            rewards=rewards,
            delivered=total_delivered,
            dropped=total_dropped,
            avg_latency=last_avg_latency,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Client is optional — if no API key, the fallback greedy agent runs
    client: Optional[OpenAI] = None
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    for task_type in TASK_TYPES:
        run_task(task_type, client)


if __name__ == "__main__":
    main()
