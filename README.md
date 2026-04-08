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

An OpenEnv environment for training RL agents to optimize network routing. Agents learn to select between different network paths (Fiber, Copper, Satellite) to minimize latency and packet loss under dynamic network conditions.

**Features:**
- 🌐 **Network Routing Simulation** - Choose between 3 network paths with different characteristics
- 🎲 **Dynamic Chaos Mode** - Network conditions change unpredictably every 5 steps
- 🎯 **LLM Judge System** - Evaluate agent actions with configurable personas (lenient/standard/strict)
- 📈 **Curriculum Controller** - Automatic difficulty progression based on agent mastery
- 🤖 **Multi-Backend LLM Client** - Supports OpenAI, HuggingFace, and Anthropic APIs
- 🚀 **GRPO Training Template** - Ready-to-use training script with TRL + vLLM
- 📊 **Reward Visualization** - Built-in plotting utilities for training analysis

## How It Works

The agent must route network traffic by selecting one of three paths at each step:

| Path | ID | Normal Latency | Packet Loss | Behavior |
|------|-----|----------------|-------------|----------|
| **Fiber** | 0 | 10ms | 0.01 | Best performance, but degrades to 90ms + 0.10 loss during chaos |
| **Copper** | 1 | 45ms | 0.02 | Consistent medium performance |
| **Satellite** | 2 | 500ms | 0.00 | High latency but zero packet loss |

**Chaos Mode**: Periodically (interval depends on difficulty), Fiber becomes the worst option. Agents must learn to detect congestion and switch paths.

**Progressive Difficulty** (Curriculum Learning):
- **Easy**: Chaos every 10 steps, 20-step episodes, predictable
- **Medium**: Chaos every 6-7 steps, 50-step episodes  
- **Hard**: Chaos every 3 steps, 100-step episodes, ±5ms noise added
- Automatically advances as agent improves

**Reward Function**: Base + Shaping
- Base: `(100.0 / (latency + 1)) - (packet_loss * 50)`
- Switching penalty: -0.5 (discourages thrashing)
- Anticipation bonus: +1.0 (rewards predicting chaos)
- Consistency bonus: +0.3 (rewards stable performance)

## Quick Start

### Basic Usage

```python
from smart_router import SmartRouterEnv, SmartRouterAction

# Connect to the environment
with SmartRouterEnv(base_url="http://localhost:8000") as env:
    # Reset to start
    result = env.reset()
    print(f"Latency: {result.observation.latency_ms}ms")
    print(f"Packet Loss: {result.observation.packet_loss}")
    print(f"Congested: {result.observation.is_congested}")
    
    # Select Fiber path (ID=0)
    result = env.step(SmartRouterAction(path_selection=0))
    print(f"Reward: {result.reward:.2f}")
    
    # Detect congestion and switch to Copper
    if result.observation.is_congested:
        result = env.step(SmartRouterAction(path_selection=1))
```

### Run the Example Agent

The repository includes a simple adaptive agent that switches between Fiber and Copper based on congestion:

```bash
# Start the environment server
uv run server

# In another terminal, run the agent
python run_agent.py
```

**Agent Logic**:
```python
is_congested = observation.is_congested
action = 1 if is_congested else 0  # Copper if congested, else Fiber
```

### Using Docker

```python
from smart_router import SmartRouterEnv

# Automatically start container and connect
env = SmartRouterEnv.from_docker_image("smart_router-env:latest")
try:
    result = env.reset()
    result = env.step(SmartRouterAction(path_selection=0))
    print(f"Reward: {result.reward:.2f}")
finally:
    env.close()
```

Build the Docker image first:
```bash
docker build -t smart_router-env:latest -f server/Dockerfile .
```

## Environment Details

### Action

**SmartRouterAction**
- `path_selection` (int, 0-2): Network path to use
  - `0` = Fiber
  - `1` = Copper  
  - `2` = Satellite

### Observation

**SmartRouterObservation**
- `latency_ms` (float): Network latency in milliseconds
- `packet_loss` (float): Packet loss rate (0.0-1.0)
- `is_congested` (bool): True if latency > 60ms

### Reward

Base reward: **`(100.0 / (latency + 1)) - (packet_loss * 50)`**

**Reward Shaping** (helps agent learn better strategies):
- **Switching penalty**: -0.5 when changing paths (simulates BGP convergence cost)
- **Anticipation bonus**: +1.0 for switching to Copper before chaos hits
- **Consistency bonus**: +0.3 for maintaining stable performance (low variance)

Examples:
- Fiber (normal): latency=10ms, loss=0.01 → base reward ≈ 8.59
- Fiber (chaos): latency=90ms, loss=0.10 → base reward ≈ -3.90
- Copper: latency=45ms, loss=0.02 → base reward ≈ 1.17
- Satellite: latency=500ms, loss=0.00 → base reward ≈ 0.20

**Curriculum Effects**:
- At higher difficulties, latency gets random noise (±5ms)
- Episode length increases from 20 steps (easy) to 100 steps (hard)
- Chaos becomes more frequent (every 10 steps → every 3 steps)

## Advanced Features

This environment includes production-ready features for RL agent training.

### Curriculum Learning (Integrated)

The environment automatically adjusts difficulty based on agent performance:

```python
# Curriculum is integrated into the environment automatically
from smart_router import SmartRouterEnv

env = SmartRouterEnv(base_url="http://localhost:8000")

# First reset: Easy (chaos every 10 steps, 20 step episodes)
result = env.reset()
print(f"Chaos interval: {result.info['chaos_interval']}")  # 10
print(f"Max steps: {result.info['max_steps']}")  # 20

# As agent succeeds, curriculum increases difficulty
# After many successful episodes: Hard (chaos every 3 steps, 100 step episodes)
```

**How it works:**
- Tracks agent success rate over recent episodes
- **Easier levels**: Chaos every 10 steps, shorter 20-step episodes
- **Harder levels**: Chaos every 3 steps, longer 100-step episodes, latency noise
- Automatically advances through: warmup → beginner → intermediate → advanced → expert
- Fast-track: 90%+ success rate advances immediately

**Curriculum stats available in step info:**
```python
result = env.step(SmartRouterAction(path_selection=0))
print(result.info['difficulty'])  # 0.0-1.0
print(result.info['curriculum_tier'])  # "warmup", "beginner", etc.
```

### LLM Judge (Offline Evaluation Only)

**Important**: The LLM judge is for **offline evaluation**, not online training.

Evaluate trained agent policies:

```python
from server.llm_client import LLMClient
from server.judge import LLMJudge

# Set up LLM backend (reads LLM_BACKEND env var)
llm = LLMClient()
judge = LLMJudge(llm)

# Run episode with your trained agent
agent = load_trained_agent("checkpoints/routing-agent")
env = SmartRouterEnv(base_url="http://localhost:8000")
result = env.reset()
history = []

for step in range(100):
    action = agent.predict(result.observation)
    result = env.step(SmartRouterAction(path_selection=action))
    
    # Evaluate action quality offline (not used for training)
    score, feedback = judge.evaluate(
        action=f"path_selection: {action}",
        observation=f"latency: {result.observation.latency_ms}ms",
        context={
            "task_description": "Optimize network routing",
            "goal": "Minimize latency and packet loss",
        },
        history=history,
        persona="strict"
    )
    
    history.append({
        "step": step,
        "action": action,
        "judge_score": score,
        "feedback": feedback
    })

# Analyze results
avg_judge_score = sum(h['judge_score'] for h in history) / len(history)
print(f"Average judge score: {avg_judge_score:.2f}")
```

**Why offline only?**
- Each judge call takes 200-500ms (too slow for training)
- Cost: $0.10-1.00 per 100-step episode with API providers
- LLM outputs vary, creating noisy gradients

See **[JUDGE_USAGE.md](JUDGE_USAGE.md)** for detailed examples and best practices.

**Configure LLM Backend:**

```bash
# Use Anthropic Claude (recommended for production)
export LLM_BACKEND=anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Use OpenAI-compatible (vLLM, OpenAI)
export LLM_BACKEND=openai
export LLM_BASE_URL=http://localhost:8001/v1
export LLM_MODEL=gpt-3.5-turbo

# Use HuggingFace Inference API
export LLM_BACKEND=hf
export HF_TOKEN=hf_...
export LLM_MODEL=Qwen/Qwen3-14B
```

### GRPO Training

Train agents using Group Relative Policy Optimization with the included template:

```bash
# Install training dependencies
pip install -e ".[train]"

# Start environment server
uv run server

# Train agent (in separate terminal)
python train_template.py \
  --model-id Qwen/Qwen3-0.6B \
  --env-url http://localhost:8000 \
  --dataset-size 50 \
  --num-generations 8 \
  --output-dir outputs/routing-agent
```

**Training features:**
- Multi-turn episode support
- Conversation history tracking
- LoRA fine-tuning
- Automatic checkpoint saving
- CSV reward logging
- HuggingFace Hub integration

### Reward Visualization

Plot training progress from reward logs:

```bash
python plot_rewards.py outputs/routing-agent/reward_log.csv
```

Generates a plot with:
- Per-episode rewards
- Rolling average (10 episodes)
- Trend line with slope
- Summary statistics

## Deploying to Hugging Face Spaces

Deploy your environment to HuggingFace Spaces using the OpenEnv CLI:

```bash
# From the environment directory
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate the OpenEnv environment
2. Prepare a custom build for HuggingFace Docker space
3. Upload to HuggingFace

### Prerequisites

Authenticate with HuggingFace (the command will prompt if needed):
```bash
huggingface-cli login
```

### Options

- `--directory`, `-d`: Directory containing the environment (default: current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name'
- `--base-image`, `-b`: Base Docker image to use
- `--private`: Deploy as private space

### Examples

```bash
# Push to your personal namespace
openenv push

# Push to a specific repository
openenv push --repo-id my-org/smart-router

# Push as a private space
openenv push --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint

## Advanced Usage

### Connecting to an Existing Server

If you have a Smart Router environment server running elsewhere:

```python
from smart_router import SmartRouterEnv

# Connect to existing server (won't stop it on close)
env = SmartRouterEnv(base_url="https://my-server.com")
result = env.reset()
result = env.step(SmartRouterAction(path_selection=0))
```

### Context Manager Pattern

The client supports context manager usage for automatic connection management:

```python
from smart_router import SmartRouterAction, SmartRouterEnv

with SmartRouterEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    
    # Run episode
    for step in range(100):
        # Simple strategy: use Fiber unless congested
        action = 1 if result.observation.is_congested else 0
        result = env.step(SmartRouterAction(path_selection=action))
        
        if result.done:
            break
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent Sessions

The server supports multiple concurrent WebSocket connections. To enable this, modify `server/app.py`:

```python
app = create_app(
    SmartRouterEnvironment,  # Pass class, not instance
    SmartRouterAction,
    SmartRouterObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then run multiple clients simultaneously:

```python
from smart_router import SmartRouterAction, SmartRouterEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with SmartRouterEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        total_reward = 0
        
        for i in range(100):
            action = 1 if result.observation.is_congested else 0
            result = env.step(SmartRouterAction(path_selection=action))
            total_reward += result.reward
            
            if result.done:
                break
                
        return client_id, total_reward

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
    
for client_id, reward in results:
    print(f"Client {client_id}: Total reward = {reward:.2f}")
```

## Development & Testing

### Direct Environment Testing

Test the environment logic without starting the HTTP server:

```bash
python server/smart_router_environment.py
```

This verifies:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly
- Chaos mode triggers appropriately

### Running Locally

Run the server locally for development:

```bash
# With auto-reload
uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

# Or using the entry point
uv run server

# With custom port
uv run server --port 8001
```

## Installation

```bash
# Basic usage
pip install -e .

# With training support (GRPO + vLLM)
pip install -e ".[train]"

# With evaluation tools
pip install -e ".[eval]"

# Development mode
pip install -e ".[dev]"
```

## Inference Script

For benchmarking and competitions, use the standardized inference script:

```bash
# Set up environment
export HF_TOKEN=hf_your_token_here
export LOCAL_IMAGE_NAME=smart_router-env:latest
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

# Run inference
python inference.py
```

**Output format:**
```
[START] task=routing env=smart_router model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=0(Fiber) reward=8.09 done=false error=null
[STEP] step=2 action=1(Copper) reward=1.67 done=false error=null
...
[END] success=true steps=100 score=0.847 rewards=8.09,1.67,...
```

See **[INFERENCE_README.md](INFERENCE_README.md)** for detailed usage.

## Project Structure

```
smart_router/
├── .dockerignore                    # Docker build exclusions
├── .gitignore                       # Git exclusions
├── __init__.py                      # Module exports
├── README.md                        # This file
├── INFERENCE_README.md              # Inference script documentation
├── JUDGE_USAGE.md                   # LLM Judge usage guide (offline evaluation)
├── ANALYSIS_REWARD_CURRICULUM.md   # Design analysis and recommendations
├── CHANGES_SUMMARY.md               # Summary of recent improvements
├── inference.py                     # Standardized inference script
├── openenv.yaml                     # OpenEnv manifest
├── pyproject.toml                   # Project metadata and dependencies
├── requirements.txt                 # Generated dependencies
├── client.py                        # SmartRouterEnv client
├── models.py                        # Action and Observation models
├── run_agent.py                     # Example adaptive agent
├── train_template.py                # GRPO training script template
├── plot_rewards.py                  # Reward visualization utilities
└── server/
    ├── __init__.py                  # Server module exports
    ├── smart_router_environment.py  # Core environment logic
    ├── app.py                       # FastAPI application (HTTP + WebSocket)
    ├── llm_client.py                # Multi-backend LLM client (OpenAI/HF/Anthropic)
    ├── judge.py                     # LLM-based action evaluator
    ├── curriculum.py                # Automatic difficulty progression
    ├── Dockerfile                   # Container image definition
    └── requirements.txt             # Server-specific dependencies
```
