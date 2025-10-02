"""MTOB (Machine Translation from One Book) domain environment for SkyRL training.

Provides MTOBEnv class for machine translation with advisor feedback.
Kalamang->English translation using chrF score evaluation.
"""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from typing import Dict, Any
from omegaconf import DictConfig
import evaluate
from openai import OpenAI

from .config import STUDENT_INSTRUCTIONS


class MTOBEnv(BaseTextEnv):
    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        assert "original_question" in extras

        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.original_question = extras["original_question"]

        # Store pre-built reference materials from dataset
        self.reference_materials = extras.get("reference_materials", [])
        assert len(self.reference_materials) > 0, "No reference materials found"

        # Initialize evaluation metrics
        self.chrf_metric = evaluate.load("chrf")

    def _build_prompt(self, advisor_feedback: str) -> str:
        """Compose the prompt sent to the student model following MTOB format."""

        # Build the student prompt
        return STUDENT_INSTRUCTIONS.format(
            original_question=self.original_question,
            reference_materials="\n".join(self.reference_materials),
            advisor_feedback=advisor_feedback,
        )

    def call_openai(self, prompt: str, temperature: float = 0.0) -> str:
        """Call the completion endpoint using OpenAI."""
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[MTOBEnv] LLM request failed: {e}")
            return ""

    def step(self, action: str) -> BaseTextEnvStepOutput:
        """Single-step episode evaluating advisor feedback.

        The agent (`action`) provides feedback which is forwarded to the
        student model. The returned translation is scored against the ground
        truth using chrF metric.
        """
        prompt = self._build_prompt(action)
        updated_response = extract_translation(self.call_openai(prompt))
        reward = compute_translation_score(updated_response, self.ground_truth)

        metadata = {
            "updated_response": updated_response,
            "ground_truth": self.ground_truth,
        }

        return BaseTextEnvStepOutput(
            observations=[],
            reward=reward,
            done=True,
            metadata=metadata,
        )


def compute_translation_score(translation: str, ground_truth: str) -> float:
    """Compute translation quality score using chrF metric.

    Args:
        translation: The model's translation
        ground_truth: The reference translation

    Returns:
        chrF score (0-100 scale, normalized to 0-1)
    """
    if not translation or not translation.strip():
        return 0.0

    try:
        # Initialize chrF metric
        chrf_metric = evaluate.load("chrf")

        # Compute chrF score
        chrf_result = chrf_metric.compute(
            predictions=[translation.strip()], references=[ground_truth.strip()]
        )

        # chrF returns score on 0-100 scale, normalize to 0-1
        return chrf_result["score"] / 100.0

    except Exception as e:
        print(f"Error computing translation score: {e}")
        return 0.0


def extract_translation(response: str) -> str:
    """Extract the translation from the model response.

    For MTOB, we typically expect the entire response to be the translation,
    but we'll clean it up by removing any extra formatting.
    """
    if not response:
        return ""

    # Remove common prefixes that models might add
    response = response.strip()

    # Extract translation from response
    try:
        translation = response.split("Translation: ")[1].strip()
    except Exception:
        return ""

    # Remove quotes if the entire response is quoted
    if translation.startswith('"') and translation.endswith('"'):
        translation = translation[1:-1]
    elif translation.startswith("'") and translation.endswith("'"):
        translation = translation[1:-1]

    return translation.strip()
