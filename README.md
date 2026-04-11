---
title: Smart Router Environment
emoji: ⌨️
colorFrom: indigo
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Smart Router Environment

An OpenEnv environment for training RL agents to optimize network routing in a realistic multi-hop IP network. The agent acts as a specific router (R2) and makes per-hop forwarding decisions for packets arriving at its interfaces, while background traffic independently loads links and random burst events create dynamic congestion.

**Features:**
- 🌐 **Multi-Hop Network Graph** — 5-to-14 node topologies with real router-to-router links and per-link latency/capacity/loss characteristics
- 📦 **Packet Simulation** — Each packet has source, destination, priority (low/medium/high), TTL, and a visited-node list for cycle detection
- 🚦 **Background Traffic Engine** — Independent flows and burst events load links, creating realistic congestion the agent must route around
- 📋 **Per-Router Forwarding Table (FIB)** — Dijkstra-computed shortest-path tables are visible to the agent, like real routing protocols
- 🔄 **Dynamic Topology** — Network graph grows in size and complexity as curriculum difficulty increases (5 nodes at warmup → 14 nodes at expert)
- ⚠️ **Penalty System** — Invalid hops (non-neighbour index), routing loops (cycle in visited), and TTL expiry all produce negative rewards
- 📈 **Curriculum Controller** — Automatic difficulty progression based on agent mastery
- 🎯 **LLM Judge System** — Evaluate agent actions with configurable personas (lenient/standard/strict)
- 🤖 **Multi-Backend LLM Client** — Supports OpenAI, HuggingFace, and Anthropic APIs
- 🚀 **GRPO Training Template** — Ready-to-use training script with TRL + vLLM

---

## How It Works

### The Agent's Role

The agent IS router **R2**. Every step, a packet arrives at R2 that needs to be forwarded toward its destination. The agent inspects the packet and the current state of its outgoing links, then picks a next-hop by index.

```
episode: generate_topology(difficulty) → build FIB → init TrafficEngine
              ↓
step:  packet arrives at R2
       agent picks next_hop_index (integer index into link_states)
       env: validate → simulate_remainder_via_FIB → TrafficEngine.advance()
       reward = delivery_quality + progress + congestion_penalty + penalty_flags
       next observation (next packet)
```

### Network Topology (scales with difficulty)

| Tier | Difficulty | Nodes | Links | Background flows |
|------|-----------|-------|-------|-----------------|
| Warmup | 0.0–0.25 | 5 | 6 | 2–4 |
| Beginner | 0.25–0.40 | 7 | 10 | 4–7 |
| Intermediate | 0.40–0.60 | 9 | 14 | 7–12 |
| Advanced | 0.60–0.80 | 11 | 20 | 12–18 |
| Expert | 0.80–1.00 | 14 | 28 | 18–30 |

R2 is always placed at a high-betweenness-centrality position so that most traffic flows through it.

### Packets

Each packet has:
- **src / dst** — origin and destination router names
- **priority** — 0 (low), 1 (medium), 2 (high); affects reward magnitude
- **TTL** — time-to-live; decrements per hop, packet dropped at 0
- **visited** — list of routers already traversed; picking one = cycle = penalty

### Background Traffic & Bursts

The `TrafficEngine` maintains concurrent flows between random router pairs. Each step:
1. Flows age and expire.
2. New flows spawn to maintain the target count (scales with difficulty).
3. With 5–20% probability (scales with difficulty), a burst event fires — a random path gets 300–600 Mbps of spike load for 3–10 steps.
4. Per-link loads are recomputed; this changes the effective latency and utilisation the agent observes.

### Forwarding Information Base (FIB)

At episode start, Dijkstra is run from every node to compute shortest-path next-hops. The FIB is exposed to the agent as `fib: List[FIBEntry]` — the same information a real router's routing table provides. The agent can follow it or deviate (e.g., to load-balance around congestion).

### Reward Function

| Situation | Base Reward | Priority Multiplier |
|-----------|-------------|---------------------|
| Delivered, low total latency | +10 to +15 | × 0.5 / 1.0 / 2.0 |
| Delivered, high latency | +10 | × priority |
| Remainder path lost packet | −3 | × priority |
| Invalid index (out of range) | −10 | × priority |
| Cycle detected (in visited) | −8 | × priority |
| TTL expired | −5 | × priority |
| Intentional drop (−1) | −1 | × priority |
| Progress toward dst (+1 hop closer) | +2 | × priority |
| Congestion penalty (util > 70 %) | up to −4 | × priority |
| Load-balance bonus (smart FIB deviation) | +1.5 | × priority |
| Bad deviation (chose congested over clear FIB) | −1.0 | × priority |
| Consistency bonus (low reward variance) | +0.3 | — |

---

## Quick Start

### Run the Example Agent (no server needed)

```bash
# Start the environment server
uv run server

# In another terminal, run the reactive agent
python run_agent.py
```

The agent follows the FIB suggestion unless the link is congested, then picks the least-loaded valid alternative. Example output:

```
Step    1 | idx=1(R3) 12.1ms util=0.21          | reward=  +9.80 | pkt=a3f1b2 R0->R9 pri=2 ttl=21 | delivered=True  | flows=8 net=0.22
Step    2 | idx=0(R1) 45.2ms util=0.83           | reward= -12.40 | pkt=c7d9e1 R4->R8 pri=2 ttl=12 | delivered=False | flows=9 net=0.35 | ERR:REMAINDER_FAIL:STOCHASTIC_LOSS
Step    3 | idx=2(R5) 8.4ms util=0.14            | reward= +14.20 | pkt=f0a2b3 R1->R9 pri=2 ttl=21 | delivered=True  | flows=8 net=0.27
```

### Run Inference (primary evaluation)

```bash
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=google/gemma-4-31B-it

python inference.py
```

`inference.py` is the primary evaluation file. It runs all five task scenarios in sequence:

| Task | Topology | Focus |
|------|----------|-------|
| `routing-basic` | 5 nodes, low load | Baseline forwarding |
| `routing-congested` | 9 nodes, heavy background flows | Congestion avoidance |
| `routing-burst` | 9 nodes, frequent bursts | Burst reaction |
| `routing-multihop` | 11 nodes, long paths | Multi-hop planning |
| `routing-expert` | 14 nodes, max load + bursts | Full complexity |

**Output format:**
```
[START] task=routing-expert env=smart_router model=gemma-4-31B nodes=14 links=28 agent=R2
[STEP] step=1 action=1(R3) reward=+12.40 done=false packet=a3f1 src=R0 dst=R9 pri=high ttl=21 fib=R3 delivered=true total_lat=67.2ms link_lat=10.3ms util=0.28 active_flows=22 net_util=0.61 error=null
[STEP] step=2 action=0(R1) reward=-16.00 done=false packet=b7c2 src=R4 dst=R8 pri=high ttl=3 fib=R1 delivered=None total_lat=None link_lat=None util=None active_flows=23 net_util=0.63 error=CYCLE:R1_already_in_visited
[END] success=true steps=50 score=0.712 rewards=12.40,-16.00,... delivered=38 dropped=12 delivery_rate=0.760 avg_latency=71.3ms
```

### Direct API Usage

```python
from smart_router import SmartRouterEnv, SmartRouterAction

with SmartRouterEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    obs = result.observation

    print(f"Agent router: {obs.agent_router}")          # R2
    print(f"Packet: {obs.packet.src} → {obs.packet.dst}")
    print(f"Priority: {obs.packet.priority}")
    print(f"TTL: {obs.packet.ttl}")
    print(f"Visited: {obs.packet.visited}")

    # Show available links
    for link in obs.link_states:
        print(f"  [{link.index}] {link.neighbor}: {link.latency_ms}ms util={link.utilization:.2f}")

    # Show FIB suggestion
    for entry in obs.fib:
        if entry.destination == obs.packet.dst:
            print(f"FIB suggests: {entry.next_hops}")

    # Forward to neighbour at index 1
    result = env.step(SmartRouterAction(next_hop_index=1))
    print(f"Reward: {result.reward:.2f}")
```

---

## Environment API

### Action

**`SmartRouterAction`**
- `next_hop_index` (int, ≥ −1):
  - `−1` = intentionally drop the packet (mild penalty)
  - `0 … N−1` = forward to the neighbour at that index in `link_states`
  - Out-of-range index → large penalty, packet dropped

### Observation

**`SmartRouterObservation`**

| Field | Type | Description |
|-------|------|-------------|
| `agent_router` | str | The router the agent controls (always `"R2"`) |
| `packet` | PacketInfo | Current packet awaiting forwarding |
| `link_states` | List[LinkState] | One entry per outgoing link |
| `fib` | List[FIBEntry] | Forwarding table (Dijkstra suggestions) |
| `queue_size` | int | Packets pending at agent router |
| `active_flow_count` | int | Background flows currently active |
| `network_utilization` | float | Average utilisation across all links |

**`PacketInfo`**

| Field | Type | Description |
|-------|------|-------------|
| `packet_id` | str | Unique identifier |
| `src` | str | Originating router |
| `dst` | str | Destination router |
| `priority` | int | 0=low, 1=medium, 2=high |
| `ttl` | int | Remaining hops before expiry |
| `hops_taken` | int | Hops completed so far |
| `visited` | List[str] | Routers already traversed |

**`LinkState`**

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Action index to use |
| `neighbor` | str | Router this link connects to |
| `latency_ms` | float | Current effective latency (base + queuing delay) |
| `utilization` | float | 0.0–1.0 fraction of capacity used |
| `queue_depth` | int | Approximate packets queued |
| `is_congested` | bool | True when utilization > 0.75 |

**`FIBEntry`**

| Field | Type | Description |
|-------|------|-------------|
| `destination` | str | Destination router name |
| `next_hops` | List[str] | `[primary, alt]` next-hop router names |

---

## Advanced Features

### Curriculum Learning (Integrated)

The environment automatically adjusts difficulty based on agent performance:

```python
result = env.step(SmartRouterAction(next_hop_index=0))
print(result.info['difficulty'])        # 0.0–1.0
print(result.info['curriculum_tier'])   # "warmup", "beginner", …, "expert"
print(result.info['graph_nodes'])       # 5, 7, 9, 11, or 14
print(result.info['active_flows'])      # background flows this step
print(result.info['packets_delivered']) # cumulative delivered this episode
print(result.info['avg_delivery_latency_ms'])  # cumulative average
```

Tier progression: **warmup → beginner → intermediate → advanced → expert**

Fast-track: 90 %+ success in 3 consecutive episodes advances immediately.

### LLM Judge (Offline Evaluation Only)

The judge is for offline evaluation, not online training. See `server/judge.py` for usage.

### GRPO Training

Train agents using Group Relative Policy Optimization:

```bash
pip install -e ".[train]"
uv run server

python train_template.py \
  --model-id Qwen/Qwen3-0.6B \
  --env-url http://localhost:8000 \
  --dataset-size 50 \
  --num-generations 8 \
  --output-dir outputs/routing-agent
```

### Reward Visualization

```bash
python plot_rewards.py outputs/routing-agent/reward_log.csv
```

---

## Deploying to Hugging Face Spaces

```bash
openenv push
# or
openenv push --repo-id my-org/smart-router --private
```

After deployment:
- **Web Interface** — `/web`
- **API Documentation** — `/docs`
- **Health Check** — `/health`
- **WebSocket** — `/ws`

---

## Project Structure

```
smart_router/
├── __init__.py                      # Module exports
├── README.md                        # This file
├── inference.py                     # Primary evaluation script (5 task types)
├── run_agent.py                     # Reactive greedy example agent
├── client.py                        # SmartRouterEnv client (WebSocket)
├── models.py                        # Action / Observation data models
├── train_template.py                # GRPO training script template
├── plot_rewards.py                  # Reward visualization utilities
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Project metadata and dependencies
└── server/
    ├── smart_router_environment.py  # Core environment (reset/step/reward)
    ├── topology.py                  # Network graph blueprints + FIB computation
    ├── traffic_engine.py            # Background traffic flows and burst events
    ├── curriculum.py                # Automatic difficulty progression
    ├── app.py                       # FastAPI application (HTTP + WebSocket)
    ├── judge.py                     # LLM-based action evaluator (offline)
    ├── llm_client.py                # Multi-backend LLM client
    └── Dockerfile                   # Container image definition
```

## Installation

```bash
# Basic usage
pip install -e .

# With training support (GRPO + vLLM)
pip install -e ".[train]"

# With evaluation tools
pip install -e ".[eval]"
```
