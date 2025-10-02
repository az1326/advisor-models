"""Math solutions domain environment for SkyRL training.

Provides MathSolutionsEnv class for math solution writing with advisor feedback.
Evaluates generated solutions for style alignment.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Any, Dict, List

from omegaconf import DictConfig
from openai import OpenAI

from .config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    STYLE_JUDGE_SYSTEM_PROMPT,
    STYLE_JUDGE_PROMPT,
)


class MathSolutionsEnv(BaseTextEnv):
    """Environment for math solution writing with advisor feedback using 2-step flow."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        # Required fields
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "judge_criteria" in extras["reward_spec"], (
            "judge_criteria field is required"
        )
        assert "original_question" in extras, "original_question field is required"
        assert "student" in extras, "student field is required"
        assert "ground_truth_answer" in extras, "ground_truth_answer field is required"

        self.original_question = extras["original_question"]
        self.student = extras["student"]
        self.ground_truth_answer = extras["ground_truth_answer"]
        self.judge_criteria = extras["reward_spec"]["judge_criteria"]

    def _build_student_prompt(self, advisor_feedback: str) -> List[Dict[str, str]]:
        """Build prompt for student model to solve the math problem."""
        user_context = STUDENT_INSTRUCTION.format(
            problem=self.original_question,
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
            print(f"[MathSolutionsEnv] LLM request failed: {e}")
            return ""

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step: advisor feedback -> student model -> solution evaluation."""
        # Step 1: Advisor provides feedback (this is the action)
        advisor_feedback = action

        # Step 2: Build prompt for student model with advisor feedback
        messages = self._build_student_prompt(advisor_feedback)

        # Step 3: Get solution from student model
        solution_response = self.call_openai(messages, temperature=0.0)

        # Step 4: Compute reward
        total_reward = self._compute_reward(solution_response)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=total_reward,
            done=True,
            metadata={
                "advisor_feedback": advisor_feedback,
                "updated_response": solution_response,
                "ground_truth": self._format_task_info(),
            },
        )

    def _compute_reward(self, solution: str) -> float:
        """Compute style matching reward using LLM-as-a-judge."""
        try:
            judge_prompt = STYLE_JUDGE_PROMPT.format(
                judge_criteria=self.judge_criteria,
                solution=solution,
            )

            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": STYLE_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
            )

            judge_response = response.choices[0].message.content.strip()

            # Parse the three-way response: ACCEPT=1.0, PARTIAL=0.4, REJECT=0.0
            if "ACCEPT" in judge_response:
                return 1.0
            elif "PARTIAL" in judge_response:
                return 0.4
            else:  # REJECT or any other response
                return 0.0

        except Exception as e:
            print(f"Error computing style reward: {e}")
            return 0.0

    def _format_task_info(self) -> str:
        """Format task information in natural language."""
        return f"{self.student} prefers:\n{self.judge_criteria}\nAnswer was {self.ground_truth_answer}"
