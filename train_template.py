"""
GRPO Training Template for OpenEnv environments.

Adapt this template for your specific environment by:
1. Importing your environment class and action/observation types
2. Customizing the SYSTEM_PROMPT for your task
3. Adjusting format_observation() for your observation structure
4. Modifying reward functions as needed
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

# Help PyTorch reuse fragmented GPU memory
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")

from datasets import Dataset
from peft import LoraConfig

# CUSTOMIZE: Import your environment
from smart_router import SmartRouterAction, SmartRouterEnv
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from trl.experimental.openenv import generate_rollout_completions

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# CUSTOMIZE: Your task-specific system prompt
SYSTEM_PROMPT = """You are an AI assistant completing tasks in an interactive environment.

Output ONE action per turn. Be clear and concise.

Follow a systematic approach:
1. Understand the current state
2. Plan your next action
3. Execute the action
4. Verify the result"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO training for OpenEnv agent")
    parser.add_argument(
        "--model-id", default="Qwen/Qwen3-0.6B", help="Agent model to fine-tune"
    )
    parser.add_argument(
        "--env-url", default="http://localhost:8000", help="OpenEnv server URL"
    )
    parser.add_argument(
        "--dataset-size", type=int, default=50, help="Number of training episodes"
    )
    parser.add_argument(
        "--max-turns", type=int, default=15, help="Max turns per episode"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Max tokens per response"
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="GRPO generations (8+ recommended)",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="Max training steps (-1 = auto)"
    )
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-repo", default=None, help="HF Hub repo")
    parser.add_argument(
        "--vllm-mode", choices=("colocate", "server"), default="colocate"
    )
    parser.add_argument("--vllm-server-url", default="http://localhost:8001")
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--report-to", default="none", choices=("tensorboard", "wandb", "none")
    )
    parser.add_argument("--reward-log", default="reward_log.csv")
    return parser.parse_args()


# CUSTOMIZE: Format your observation into text
def format_observation(obs) -> str:
    """Convert observation to agent-readable text."""
    # Example for SmartRouterObservation
    echoed = getattr(obs, "echoed_message", "")
    length = getattr(obs, "message_length", 0)
    return f"Echoed: {echoed}\nLength: {length}"


def format_history(history: list[dict]) -> str:
    """Format conversation history."""
    if not history:
        return ""
    lines = ["PREVIOUS ACTIONS:"]
    for entry in history:
        action = entry["action"]
        observation = entry["observation"]
        if len(observation) > 200:
            observation = observation[:200] + "..."
        lines.append(f"→ {action}")
        lines.append(f"  Result: {observation}")
    return "\n".join(lines)


def apply_chat_template(tokenizer, messages):
    """Apply chat template with fallback."""
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )


def rollout_once(
    trainer: GRPOTrainer,
    env,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    max_turns: int,
) -> dict[str, list]:
    """Run one episode and return accumulated tokens + rewards."""
    result = env.reset()
    observation = result.observation

    prompt_ids: list[int] = []
    completion_ids: list[int] = []
    logprobs: list[float] = []
    step_rewards: list[float] = []
    conversation_history: list[dict] = []

    MAX_TOTAL_TOKENS = 4096

    for _turn in range(max_turns):
        if result.done or len(completion_ids) >= MAX_TOTAL_TOKENS:
            break

        history_text = format_history(conversation_history)
        obs_text = format_observation(observation)

        user_prompt = f"{history_text}\n\n{obs_text}" if history_text else obs_text

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = apply_chat_template(tokenizer, messages)

        # Generate with vLLM
        rollout_outputs = generate_rollout_completions(trainer, [prompt_text])[0]
        prompt_ids.extend(rollout_outputs["prompt_ids"])
        completion_ids.extend(rollout_outputs["completion_ids"])
        logprobs.extend(rollout_outputs["logprobs"])

        completion_text = rollout_outputs.get("text") or tokenizer.decode(
            rollout_outputs["completion_ids"], skip_special_tokens=True
        )

        # CUSTOMIZE: Parse your action from completion text
        # Example: extract message from the completion
        action_str = completion_text.strip()

        try:
            # CUSTOMIZE: Create your action object
            action = SmartRouterAction(message=action_str)
            result = env.step(action)
            reward = float(result.reward or 0.0)
            step_rewards.append(reward)
            observation = result.observation

            conversation_history.append(
                {
                    "action": action_str,
                    "observation": format_observation(observation)[:500],
                    "reward": reward,
                }
            )

            if result.done:
                break
        except Exception as e:
            logger.warning(f"Step error: {e}")
            step_rewards.append(-0.1)
            break

    total_reward = sum(step_rewards) if step_rewards else -1.0

    return {
        "prompt_ids": prompt_ids,
        "completion_ids": completion_ids,
        "logprobs": logprobs,
        "total_reward": total_reward,
    }


def reward_total(completions: list[str], **kwargs) -> list[float]:
    """Reward function for GRPO."""
    rewards = kwargs.get("total_reward")
    return [float(r) for r in rewards] if rewards else [0.0 for _ in completions]


def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("GRPO Training (OpenEnv + TRL)")
    logger.info("=" * 60)
    logger.info(f"Agent model:    {args.model_id}")
    logger.info(f"Env URL:        {args.env_url}")
    logger.info(f"Episodes:       {args.dataset_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # CUSTOMIZE: Initialize your environment
    env = SmartRouterEnv(base_url=args.env_url)

    # Dataset
    dataset = Dataset.from_dict({"prompt": ["Complete the task"] * args.dataset_size})

    # Output directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(args.output_dir or f"outputs/grpo-{timestamp}")

    # GRPO Config
    grpo_config = GRPOConfig(
        use_vllm=True,
        vllm_mode=args.vllm_mode,
        vllm_server_base_url=args.vllm_server_url
        if args.vllm_mode == "server"
        else None,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        output_dir=str(output_dir),
        max_steps=args.max_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=2,
        max_grad_norm=1.0,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_train_batch_size=1,
        generation_batch_size=args.num_generations,
        num_generations=args.num_generations,
        max_completion_length=args.max_new_tokens,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        temperature=args.temperature,
        report_to=args.report_to,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_repo,
        save_total_limit=3,
        loss_type="dapo",
        mask_truncated_completions=True,
        beta=0.01,
    )

    # Reward logging
    reward_log_path = output_dir / args.reward_log
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_counter = [0]

    with open(reward_log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "total_reward", "timestamp"])

    def _log_episode(total_r: float):
        episode_counter[0] += 1
        with open(reward_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode_counter[0], total_r, datetime.now().isoformat()])
        logger.info(f"Episode {episode_counter[0]}: reward={total_r:.2f}")

    def rollout_func(prompts: list[str], trainer: GRPOTrainer) -> dict[str, list]:
        episode_prompt_ids, episode_completion_ids, episode_logprobs, total_rewards = (
            [],
            [],
            [],
            [],
        )

        for _ in prompts:
            episode = rollout_once(
                trainer, env, tokenizer, SYSTEM_PROMPT, args.max_turns
            )
            episode_prompt_ids.append(episode["prompt_ids"])
            episode_completion_ids.append(episode["completion_ids"])
            episode_logprobs.append(episode["logprobs"])
            total_rewards.append(episode["total_reward"])
            _log_episode(episode["total_reward"])

        return {
            "prompt_ids": episode_prompt_ids,
            "completion_ids": episode_completion_ids,
            "logprobs": episode_logprobs,
            "total_reward": total_rewards,
        }

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # Trainer
    trainer = GRPOTrainer(
        model=args.model_id,
        processing_class=tokenizer,
        reward_funcs=[reward_total],
        train_dataset=dataset,
        args=grpo_config,
        rollout_func=rollout_func,
        peft_config=peft_config,
    )

    # Train
    logger.info("Starting GRPO training...")
    try:
        trainer.train()
    finally:
        env.close()

    trainer.save_model(str(output_dir))
    logger.info(f"Model saved to {output_dir}")

    if args.push_to_hub and args.hub_repo:
        trainer.push_to_hub()
        logger.info(f"Model pushed to {args.hub_repo}")


if __name__ == "__main__":
    main()
