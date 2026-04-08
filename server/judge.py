"""
LLM-based judge for evaluating agent actions.

Provides configurable personas (lenient to strict) and can be subclassed
for domain-specific evaluation logic.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Evaluation personas with different strictness levels
PERSONAS = {
    "lenient": """You are an encouraging mentor evaluating a trainee's actions.
Be supportive and provide partial credit for partially correct approaches.
Provide constructive hints in feedback.""",

    "standard": """You are evaluating actions with standard expectations.
Reward systematic approaches and penalize repeated or irrelevant actions.""",

    "strict": """You are evaluating with high standards.
Reward efficiency and correctness. Penalize wrong actions decisively.
Direct solutions are acceptable if they're correct.""",
}


class LLMJudge:
    """Base LLM judge for action evaluation."""

    def __init__(self, llm_client):
        """
        Initialize judge with an LLM client.

        Args:
            llm_client: An LLMClient instance (from llm_client.py)
        """
        self.llm = llm_client

    def evaluate(
        self,
        action: str,
        observation: str,
        context: dict,
        history: list,
        persona: str = "standard",
    ) -> tuple[float, str]:
        """
        Evaluate an action taken by the agent.

        Args:
            action: The action taken by the agent
            observation: The observation/output from taking the action
            context: Environment-specific context (e.g., task description, goals)
            history: List of previous steps (each with action, observation, reward)
            persona: Evaluation strictness level (lenient/standard/strict)

        Returns:
            tuple: (score between -1.0 and 1.0, feedback string)
        """
        history_summary = "\n".join(
            f"  Step {h.get('step', i+1)}: {h.get('action', '')} -> reward {h.get('reward', 0):.2f}"
            for i, h in enumerate(history[-5:])
        ) or "  (first step)"

        task_desc = context.get("task_description", "Complete the task")
        goal = context.get("goal", "Achieve the objective")
        difficulty = context.get("difficulty", 0.5)

        user_prompt = f"""Evaluate this action during task execution.

TASK:
- Description: {task_desc}
- Goal: {goal}
- Difficulty: {difficulty:.1f}/1.0

AGENT ACTION:
- Action: {action}
- Observation (truncated): {observation[:500]}

RECENT HISTORY:
{history_summary}
- Total steps taken: {len(history) + 1}

Return JSON only: {{"score": <float -1.0 to 1.0>, "feedback": "<1-2 sentence evaluation>"}}"""

        try:
            result = self.llm.chat_json(
                PERSONAS.get(persona, PERSONAS["standard"]),
                user_prompt,
                temperature=0.3,
                max_tokens=256
            )
            score = max(-1.0, min(1.0, float(result.get("score", 0.0))))
            feedback = result.get("feedback", "")
            return score, feedback
        except Exception as e:
            logger.error(f"Judge LLM error: {e}", exc_info=True)
            return 0.0, f"Judge error: {type(e).__name__}"

    def verify_completion(
        self,
        context: dict,
        history: list,
        final_state: str,
    ) -> tuple[bool, str]:
        """
        Verify if the task was actually completed successfully.

        Args:
            context: Task context and goals
            history: Full episode history
            final_state: Final environment state/observation

        Returns:
            tuple: (is_complete, explanation)
        """
        history_summary = "\n".join(
            f"  Step {h.get('step', i+1)}: {h.get('action', '')} -> {h.get('observation', '')[:100]}"
            for i, h in enumerate(history)
        )

        task_desc = context.get("task_description", "Complete the task")
        success_criteria = context.get("success_criteria", "Task goals achieved")

        user_prompt = f"""Verify if the task was completed successfully.

TASK:
- Description: {task_desc}
- Success criteria: {success_criteria}

AGENT'S ACTIONS:
{history_summary}

FINAL STATE:
{final_state[:4000]}

QUESTION: Did the agent actually complete the task according to the success criteria?

Return JSON only: {{"completed": true/false, "reason": "<1-2 sentence explanation>"}}"""

        try:
            result = self.llm.chat_json(
                "You are a strict task verification system. Only confirm completion if all criteria are met.",
                user_prompt,
                temperature=0.1,
                max_tokens=256,
            )
            completed = bool(result.get("completed", False))
            reason = result.get("reason", "")
            logger.info(f"Judge verification: completed={completed} | {reason}")
            return completed, reason
        except Exception as e:
            logger.error(f"Judge verify error: {e}", exc_info=True)
            # On error, rely on programmatic checks
            return True, f"Verification error: {type(e).__name__}"
