"""Math domain environment for SkyRL training.

Provides MathEnv class for math problem solving with advisor feedback.
Uses math_verify for answer extraction and verification.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any, List
from omegaconf import DictConfig
from openai import OpenAI
from math_verify import parse, verify

from .config import STUDENT_FINAL_SYSTEM_PROMPT


class MathEnv(BaseTextEnv):
    """Environment for math problem solving with advisor feedback."""

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        # Required fields
        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        assert "original_question" in extras, "original_question field is required"
        assert "original_response" in extras, "original_response field is required"

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.original_question = extras["original_question"]
        self.original_response = extras["original_response"]
        self.model = extras["model"]
        self.initial_reward = extras.get("initial_reward", None)

    def _build_prompt(self, advisor_feedback: str) -> List[Dict[str, str]]:
        """Compose the prompt sent to the student model."""
        return [
            {"role": "system", "content": STUDENT_FINAL_SYSTEM_PROMPT},
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
            print(f"[MathEnv] LLM request failed: {e}")
            return ""

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Execute one step of the environment."""
        # Build prompt with advisor feedback
        messages = self._build_prompt(action)

        # Get updated response from student model
        updated_response = self.call_openai(messages, temperature=0.0)

        # Extract answer and compute reward
        extracted_answer = extract_answer(updated_response)
        reward = compute_score(extracted_answer, self.ground_truth)

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata={
                "initial_reward": self.initial_reward,
                "updated_response": updated_response,
                "ground_truth": self.ground_truth,
            },
        )


def extract_answer(response_str: str):
    """Extract the final answer from a math response using math_verify.parse."""
    try:
        # Use math_verify to parse the answer from the response
        parsed_answer = parse(response_str, parsing_timeout=None)
        return parsed_answer
    except Exception as e:
        print(f"Error parsing answer with math_verify: {e}")
        return None


def compute_score(extracted_answer, ground_truth: str) -> float:
    """Compute score by comparing extracted answer with ground truth using math_verify.verify."""
    try:
        # Parse the ground truth
        parsed_ground_truth = parse(ground_truth, parsing_timeout=None)

        # If extracted_answer is already parsed, use it directly
        # Otherwise parse it
        if extracted_answer is None:
            return 0.0

        # Use math_verify to check if answers are equivalent
        # Order is important: verify(gold, answer)
        is_correct = verify(parsed_ground_truth, extracted_answer, timeout_seconds=None)
        return 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"Error computing score with math_verify: {e}")
        return 0.0
