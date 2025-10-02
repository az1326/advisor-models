"""Rule arena (US tax) domain environment for SkyRL training.

Provides RuleArenaEnv class for US tax calculation with advisor feedback.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any, List
from omegaconf import DictConfig
import re
import numpy as np
from openai import OpenAI
from .config import STUDENT_SYSTEM_PROMPT


class RuleArenaEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        assert "original_question" in extras
        assert "original_response" in extras

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.original_question = extras["original_question"]
        self.original_response = extras["original_response"]
        self.model = extras["model"]
        self.initial_reward = extras.get("initial_reward", None)

    def _build_prompt(self, advisor_feedback: str) -> List[Dict[str, str]]:
        """Compose the prompt sent to the student model."""

        # parse out think block
        if "</think>" in advisor_feedback:
            advisor_feedback = advisor_feedback.split("</think>")[1]

        return [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": self.original_question},
            {"role": "assistant", "content": self.original_response},
            {"role": "user", "content": advisor_feedback},
        ]

    def call_openai(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> str:
        """Call the chat completion endpoint using OpenAI client."""
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:  # pragma: no cover
            print(f"[RuleArenaEnv] LLM request failed: {e}")
            return ""

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Single-step episode evaluating advisor feedback.

        The agent (`action`) provides feedback which is forwarded to the
        student model. The returned solution is scored against the ground
        truth and a sparse reward is provided.
        """
        prompt = self._build_prompt(action)
        updated_response = self.call_openai(prompt)

        # Extract answer
        extracted_answer = extract_answer(updated_response)

        # Compute reward
        reward = compute_score(extracted_answer, self.ground_truth)

        metadata = {
            "extracted_answer": extracted_answer,
            "ground_truth": self.ground_truth,
            "updated_response": updated_response,
        }

        # Add initial_reward to metadata if available
        if self.initial_reward is not None:
            metadata["initial_reward"] = self.initial_reward

        return BaseTextEnvStepOutput(
            observations=[], reward=reward, done=True, metadata=metadata
        )


def extract_answer(response_str: str) -> str:
    """Extract the final answer from the response."""
    # Use RuleArena's exact regex pattern for tax extraction
    pattern = (
        r"The total tax (owed|overpaid) is \$((?:\d{1,3}(?:,\d{3})*|\d+)(\.\d+)?)\.?"
    )
    match = re.search(pattern, response_str)
    if match:
        status = match.group(1)  # "owed" or "overpaid"
        value = float(match.group(2).replace(",", ""))
        # Return negative value for overpaid (as per RuleArena implementation)
        return str(-value if status == "overpaid" else value)

    # Fallback: return the last line or the whole response if extraction fails
    lines = response_str.strip().split("\n")
    return lines[-1].strip() if lines else ""


def compute_score(extracted_answer: str, ground_truth: str) -> float:
    """Compute the score based."""
    if not extracted_answer:
        return 0.0

    # use numpy.isclose like RuleArena reference
    try:
        extracted_val = float(extracted_answer.replace(",", "").replace("$", ""))
        ground_truth_val = float(str(ground_truth).replace(",", "").replace("$", ""))
        # Use numpy.isclose for comparison (matches RuleArena implementation)
        return 1.0 if np.isclose(extracted_val, ground_truth_val) else 0.0
    except (ValueError, TypeError):
        return 0.0
