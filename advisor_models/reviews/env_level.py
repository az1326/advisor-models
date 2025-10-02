"""Reviews level domain environment for SkyRL training.

Provides ReviewsLevelEnv class for review writing with reading level preferences.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any, Dict, List

from omegaconf import DictConfig
from openai import OpenAI
from .config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
)


class ReviewsLevelEnv(BaseTextEnv):
    """Environment for review writing with advisor feedback using 2-step flow."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        # Required fields
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "level_criteria" in extras["reward_spec"], (
            "level_criteria is required in reward_spec field"
        )
        assert "original_question" in extras, "original_question field is required"
        assert "person" in extras, "person field is required"

        # Deserialize review task from stored metadata
        self.level_criteria = extras["reward_spec"]["level_criteria"]
        self.original_question = extras["original_question"]
        self.person = extras["person"]

    def _build_student_prompt(self, advisor_feedback: str) -> List[Dict[str, str]]:
        """Build prompt for student model to write the review."""
        user_context = STUDENT_INSTRUCTION.format(
            prompt=self.original_question,
            advisor_feedback=advisor_feedback,
        )

        return [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_context},
        ]

    def call_openai(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> str:
        """Call the chat completion endpoint using OpenAI client."""
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:  # pragma: no cover
            print(f"[ReviewsLevelEnv] LLM request failed: {e}")
            return ""

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step: advisor feedback -> student model -> review evaluation."""
        # Step 1: Advisor provides feedback (this is the action)
        advisor_feedback = action

        # Step 2: Build prompt for student model with advisor feedback
        messages = self._build_student_prompt(advisor_feedback)

        # Step 3: Get review from student model
        review_response = self.call_openai(messages, temperature=0.0)

        # Step 4: Compute reward based on length preference
        reward = self.compute_score(review_response)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={
                "advisor_feedback": advisor_feedback,
                "updated_response": review_response,
                "ground_truth": self._format_task_preferences(),
            },
        )

    def _format_task_preferences(self) -> str:
        """Format task preferences in natural language."""
        return f"Writing review for {self.person}:\n{self.level_criteria}"

    def compute_score(self, review_text: str) -> float:
        """Compute score for the review based on person's length preferences."""
        template = self.level_criteria
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a reading level evaluator."},
                    {"role": "user", "content": template.format(review=review_text)},
                ],
            )
            response = response.choices[0].message.content
            if "Yes" in response and "No" not in response:
                return 1.0
            return 0.0
        except Exception as e:
            print(f"Error computing review score: {e}")
            return 0.0
