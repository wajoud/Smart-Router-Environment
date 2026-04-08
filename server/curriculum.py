"""
Curriculum controller for progressive difficulty in training.

Tracks agent performance and automatically adjusts:
- Task difficulty
- Evaluation persona (lenient → standard → strict)
- Task type selection based on mastery

Generic implementation suitable for any OpenEnv environment.
"""

from collections import defaultdict
import logging
import os

logger = logging.getLogger(__name__)

# Mastery tracking thresholds
MASTERY_THRESHOLD = 0.7   # 70% success rate = mastered
MASTERY_WINDOW = 10       # Look at last N episodes per task type
MIN_EPISODES_FOR_MASTERY = 3  # Need at least N attempts before graduating

# Difficulty tiers
DIFFICULTY_TIERS = [
    {"name": "warmup",       "max_diff": 0.25, "min_episodes": 5,  "advance_rate": 0.6},
    {"name": "beginner",     "max_diff": 0.40, "min_episodes": 5,  "advance_rate": 0.6},
    {"name": "intermediate", "max_diff": 0.60, "min_episodes": 8,  "advance_rate": 0.65},
    {"name": "advanced",     "max_diff": 0.80, "min_episodes": 10, "advance_rate": 0.7},
    {"name": "expert",       "max_diff": 0.95, "min_episodes": 0,  "advance_rate": 1.0},
]


class CurriculumController:
    """
    Tracks agent skill across task types and drives difficulty progression.

    Progression: warmup → beginner → intermediate → advanced → expert
    Agent must sustain success rate over multiple episodes to advance.
    """

    def __init__(self):
        self.history = defaultdict(list)        # task_type → [bool, bool, ...]
        self.step_counts = defaultdict(list)    # task_type → [int, int, ...]
        self.episode_rewards = []
        self.episode_count = 0
        self._tier_index = 0
        self._tier_episodes = 0
        self._graduated = set()                  # Task types the agent has mastered

        # Allow forcing minimum difficulty for eval
        self._min_difficulty = float(os.environ.get("EVAL_MIN_DIFFICULTY", "0.0"))
        if self._min_difficulty > 0:
            for i, tier in enumerate(DIFFICULTY_TIERS):
                if tier["max_diff"] >= self._min_difficulty:
                    self._tier_index = i
                    break
            logger.info(f"Curriculum: forced min_difficulty={self._min_difficulty}, "
                        f"starting at tier={self.get_tier_name()}")

    def record(self, task_type: str, success: bool, steps: int, reward: float):
        """Record episode outcome and check for mastery graduation."""
        self.history[task_type].append(success)
        self.step_counts[task_type].append(steps)
        self.episode_rewards.append(reward)
        self.episode_count += 1
        self._tier_episodes += 1
        self._maybe_advance_tier()

        # Check if this task type is now mastered
        recent = self.history[task_type][-MASTERY_WINDOW:]
        if (len(recent) >= MIN_EPISODES_FOR_MASTERY
                and sum(recent) / len(recent) >= MASTERY_THRESHOLD):
            if task_type not in self._graduated:
                self._graduated.add(task_type)
                logger.info(
                    f"Curriculum: agent MASTERED '{task_type}' "
                    f"({sum(recent)}/{len(recent)} success rate)"
                )

    def _maybe_advance_tier(self):
        """Advance to next difficulty tier if agent is ready."""
        if self._tier_index >= len(DIFFICULTY_TIERS) - 1:
            return
        tier = DIFFICULTY_TIERS[self._tier_index]
        recent_rate = self._recent_success_rate()

        # Fast-track: 90%+ success after 3 episodes → advance immediately
        fast_track = (self._tier_episodes >= 3 and recent_rate >= 0.9)

        if not fast_track and self._tier_episodes < tier["min_episodes"]:
            return

        if recent_rate >= tier["advance_rate"]:
            logger.info(f"Curriculum: advancing from {tier['name']} "
                        f"(rate={recent_rate:.0%}, episodes={self._tier_episodes}"
                        f"{', FAST-TRACK' if fast_track else ''})")
            self._tier_index += 1
            self._tier_episodes = 0

    def _recent_success_rate(self, window: int = 10) -> float:
        """Success rate over the last `window` episodes across all task types."""
        all_results = [r for results in self.history.values() for r in results[-window:]]
        if not all_results:
            return 0.0
        return sum(all_results) / len(all_results)

    def get_skill_profile(self) -> dict:
        """Success rate per task type over last MASTERY_WINDOW episodes."""
        return {
            tt: round(sum(results[-MASTERY_WINDOW:]) / len(results[-MASTERY_WINDOW:]), 2)
            for tt, results in self.history.items()
            if results
        }

    def get_weak_spots(self) -> list[str]:
        """Task types where agent success rate is below mastery threshold."""
        profile = self.get_skill_profile()
        return [tt for tt, rate in profile.items() if rate < MASTERY_THRESHOLD]

    def get_graduated(self) -> set[str]:
        """Task types the agent has mastered."""
        return set(self._graduated)

    def get_difficulty(self) -> float:
        """
        Continuous difficulty within the current tier.

        Scales with success rate but capped by tier maximum.
        """
        tier = DIFFICULTY_TIERS[self._tier_index]
        if self.episode_count < 3 and self._min_difficulty == 0:
            return 0.15
        rate = self._recent_success_rate()

        if self._tier_index == 0:
            tier_floor = 0.1
        else:
            tier_floor = DIFFICULTY_TIERS[self._tier_index - 1]["max_diff"]

        natural = min(tier["max_diff"], tier_floor + rate * (tier["max_diff"] - tier_floor))
        return max(natural, self._min_difficulty)

    def get_tier_name(self) -> str:
        """Current difficulty tier name."""
        return DIFFICULTY_TIERS[self._tier_index]["name"]

    def get_judge_persona(self) -> str:
        """
        Judge strictness scales with difficulty:
          lenient   (< 0.4) — encouraging, gives hints
          standard  (0.4-0.7) — normal expectations
          strict    (> 0.7) — high standards, penalizes inefficiency
        """
        d = self.get_difficulty()
        if d < 0.4:
            return "lenient"
        elif d < 0.7:
            return "standard"
        return "strict"

    def get_stats(self) -> dict:
        """Full curriculum state for logging/debugging."""
        return {
            "episode_count": self.episode_count,
            "tier": self.get_tier_name(),
            "tier_episodes": self._tier_episodes,
            "difficulty": round(self.get_difficulty(), 2),
            "judge_persona": self.get_judge_persona(),
            "skill_profile": self.get_skill_profile(),
            "graduated": sorted(self._graduated),
            "weak_spots": self.get_weak_spots(),
            "avg_reward_last_10": round(
                sum(self.episode_rewards[-10:]) / max(1, len(self.episode_rewards[-10:])), 3
            ),
        }
