from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any, List
from omegaconf import DictConfig
import re
import numpy as np
from openai import OpenAI


class TemplateEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        """Load task-specific information from extras. In this example,
        we load a ground_truth value from extras["reward_spec"]["ground_truth"]
        and an original_question from extras["original_question"]."""

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        assert "original_question" in extras

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.original_question = extras["original_question"]

    def _build_prompt(self, advisor_feedback: str) -> List[Dict[str, str]]:
        """Compose the prompt sent to the student model. In this example, we
        provide a simple prompt template."""

        user_turn = (
            f"Answer the question given the advice.\n"
            f"Question:\n{self.original_question}\n\n"
            f"Advice:\n{advisor_feedback}"
        )

        return [{"role": "user", "content": user_turn}]

    def call_openai(
        self, messages: List[Dict[str, str]], temperature: float = 0.0
    ) -> str:
        """Call the chat completion endpoint using OpenAI client. In this
        example we hard-code the student model as gpt-4o-mini"""
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[TemplateEnv] LLM request failed: {e}")
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

        return BaseTextEnvStepOutput(
            observations=[], reward=reward, done=True, metadata=metadata
        )


def extract_answer(response_str: str) -> str:
    """Extract the final answer from the response. In this example, we use the pattern
    "The final answer is: {answer}" to find the answer"""
    pattern = r"The final answer is: (\d+)"
    match = re.search(pattern, response_str)
    if match:
        return match.group(1)
    return ""


def compute_score(extracted_answer: str, ground_truth: str) -> float:
    """Compute the score. In this example, we use numpy.isclose to compare the
    extracted answer and ground truth. LLM-as-a-Judge scoring can be done through
    another call to an LLM in this function, with access to judge criteria assigned
    to variables in __init__"""
    if not extracted_answer:
        return 0.0

    try:
        extracted_val = float(extracted_answer)
        ground_truth_val = float(str(ground_truth))
        return 1.0 if np.isclose(extracted_val, ground_truth_val) else 0.0
    except (ValueError, TypeError):
        return 0.0
