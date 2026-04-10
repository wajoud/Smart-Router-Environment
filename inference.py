"""
Inference Script for Smart Router Environment
==============================================

MANDATORY REQUIREMENTS:
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

- The environment runs entirely in-process - no server or Docker required

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=routing env=smart_router model=Qwen/Qwen2.5-72B-Instruct
    [STEP] step=1 action=0(Fiber) reward=8.59 done=false error=null
    [STEP] step=2 action=0(Fiber) reward=8.09 done=false error=null
    [STEP] step=3 action=1(Copper) reward=1.67 done=false error=null
    [END] success=true steps=3 score=0.847 rewards=8.59,8.09,1.67
"""

import os
import textwrap
from typing import List, Optional

from openai import OpenAI
from smart_router import SmartRouterAction, SmartRouterObservation
from server.smart_router_environment import SmartRouterEnvironment

# Configuration - following MANDATORY requirements
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

BENCHMARK = os.getenv("SMART_ROUTER_BENCHMARK", "smart_router")
MAX_STEPS = 15
TEMPERATURE = 0.7
MAX_TOKENS = 50
SUCCESS_SCORE_THRESHOLD = 0.5  # normalized score in [0, 1]

# Estimate for max possible reward (optimal strategy: ~7 reward/step)
_MAX_REWARD_PER_STEP = 7.0
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

# Define 3 task types for hackathon validation
TASK_TYPES = ["routing", "routing-congested", "routing-stable"]

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a network routing optimizer controlling path selection for network traffic.

    PATHS AVAILABLE:
    0 = Fiber:     Low latency (10ms) but degrades during congestion (90ms + high packet loss)
    1 = Copper:    Medium latency (45ms) but consistent and reliable
    2 = Satellite: High latency (500ms) but zero packet loss

    OBSERVATIONS:
    - latency_ms: Current network latency in milliseconds
    - packet_loss: Packet loss rate (0.0 = none, 1.0 = total loss)
    - is_congested: Boolean indicating if latency > 60ms

    STRATEGY:
    - Fiber is best under normal conditions (high reward ~8)
    - When congested (is_congested=true), switch to Copper immediately
    - Avoid Satellite unless absolutely necessary
    - Learn to anticipate congestion patterns and switch proactively

    OUTPUT FORMAT:
    Reply with ONLY a single digit: 0, 1, or 2
    No explanations, no quotes, just the path number.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(
    step: int,
    latency: float,
    packet_loss: float,
    is_congested: bool,
    last_action: Optional[int],
    last_reward: float,
    history: List[str],
) -> str:
    """Build the prompt for the LLM based on current observation."""
    history_block = "\n".join(history[-4:]) if history else "None"
    status = "CONGESTED" if is_congested else "Normal"

    return textwrap.dedent(
        f"""
        Step: {step}
        Status: {status}
        Latency: {latency:.1f}ms | Packet Loss: {packet_loss:.2f} | Congested: {is_congested}

        Last Action: {last_action if last_action is not None else "None"}
        Last Reward: {last_reward:.2f}

        Recent History:
        {history_block}

        Select path (0=Fiber, 1=Copper, 2=Satellite):
        """
    ).strip()


def get_model_action(
    client: OpenAI,
    step: int,
    latency: float,
    packet_loss: float,
    is_congested: bool,
    last_action: Optional[int],
    last_reward: float,
    history: List[str],
) -> int:
    """Get path selection from LLM using OpenAI client."""
    user_prompt = build_user_prompt(
        step, latency, packet_loss, is_congested, last_action, last_reward, history
    )

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

        # Parse response - extract first digit (0, 1, or 2)
        for char in text:
            if char in "012":
                return int(char)

        # Fallback strategy: Copper if congested, else Fiber
        return 1 if is_congested else 0

    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        # Fallback strategy
        return 1 if is_congested else 0


def run_task(task_name: str, client: OpenAI) -> None:
    """Run inference episode for a specific task."""
    env = SmartRouterEnvironment(task_type=task_name)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        last_action: Optional[int] = None
        last_reward = 0.0
        done = False

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            # Get action from model
            action_id = get_model_action(
                client,
                step,
                obs.latency_ms,
                obs.packet_loss,
                obs.is_congested,
                last_action,
                last_reward,
                history,
            )

            # Execute action
            result = env.step(SmartRouterAction(path_selection=action_id))
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_action = action_id
            last_reward = reward

            # Format action string for logging (no newlines)
            path_names = {0: "Fiber", 1: "Copper", 2: "Satellite"}
            action_str = f"{action_id}({path_names.get(action_id, '?')})"

            log_step(
                step=step, action=action_str, reward=reward, done=done, error=error
            )

            # Update history for next prompt
            history.append(
                f"Step {step}: {action_str} -> {obs.latency_ms:.0f}ms, reward={reward:+.2f}"
            )

            if done:
                break

        # Calculate score in (0, 1) range - strictly between 0 and 1, not inclusive
        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.001
        score = min(max(score, 0.001), 0.999)  # clamp to (0, 1) exclusive
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)

        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    """Run inference on all task types."""
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Run all 3 tasks to meet hackathon validation requirements
    for task_type in TASK_TYPES:
        run_task(task_type, client)


if __name__ == "__main__":
    main()
