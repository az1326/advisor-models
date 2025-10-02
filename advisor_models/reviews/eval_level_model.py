"""Evaluation script for reviews level advisor model via an OpenAI-compatible endpoint.

Example usage:
    python advisor_models/reviews/eval_level_model.py \
        --advisor_api_base http://127.0.0.1:8000/v1 \
        --advisor_model advisor_model \
        --dataset_path data/reviews/validation_level.parquet \
        --student_model gpt-4o-mini
"""

import argparse
import json
import os
import pandas as pd
from typing import Dict, List, Any
from tqdm import tqdm
import numpy as np
import litellm
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    LEVEL_CRITERIA,
    LEVEL_PEOPLE,
)


class ReviewsLevelEvaluator:
    """Evaluator for reviews level advisor model via an OpenAI-compatible endpoint."""

    def __init__(
        self,
        advisor_model: str,
        advisor_api_base: str,
        student_model: str = "gpt-4o-mini",
    ):
        """Initialize evaluator to call a remote advisor endpoint."""
        self.advisor_model = advisor_model
        self.advisor_api_base = advisor_api_base
        self.student_model = student_model
        self.openai_client = OpenAI()

        # Initialize Anthropic client if using Claude models
        if "claude" in self.student_model.lower():
            self.anthropic_client = OpenAI(
                base_url="https://api.anthropic.com/v1",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )

    def generate_advisor_feedback(self, prompt: List[Dict[str, str]]) -> str:
        """Generate advisor feedback by calling the configured OpenAI-compatible endpoint."""
        try:
            response = litellm.completion(
                model=self.advisor_model,
                messages=prompt,
                temperature=0.0,
                api_base=self.advisor_api_base,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating advisor feedback: {e}")
            return ""

    def get_student_response(self, advisor_feedback: str, review_prompt: str) -> str:
        """Get review from student model using advisor feedback."""
        user_context = STUDENT_INSTRUCTION.format(
            prompt=review_prompt,
            advisor_feedback=advisor_feedback,
        )

        messages = [
            {"role": "system", "content": STUDENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_context},
        ]

        temperature = 0
        if "gpt-5" in self.student_model:
            temperature = 1.0

        try:
            if "claude" in self.student_model.lower():
                # Use Anthropic client for Claude models
                response = self.anthropic_client.chat.completions.create(
                    model=self.student_model,
                    messages=messages,
                    temperature=temperature,
                )
            else:
                # Use OpenAI client for other models
                response = self.openai_client.chat.completions.create(
                    model=self.student_model,
                    messages=messages,
                    temperature=temperature,
                )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error getting student response: {e}")
            return ""

    def compute_reading_level_score(self, review_text: str, person: str) -> float:
        """Compute reading level appropriateness score using evaluation model."""
        if person not in LEVEL_CRITERIA:
            print(f"Warning: No reading level prompt for person {person}")
            return 0.0

        template = LEVEL_CRITERIA[person]
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are a reading level evaluator."},
                    {"role": "user", "content": template.format(review=review_text)},
                ],
            )
            response_text = response.choices[0].message.content
            if "Yes" in response_text and "No" not in response_text:
                return 1.0
            return 0.0
        except Exception as e:
            print(f"Error computing reading level score: {e}")
            return 0.0

    def process_single_example(self, idx_row_tuple):
        """Process a single example - designed for multithreading."""
        idx, row = idx_row_tuple
        try:
            # Extract data from row
            prompt = row["prompt"]
            original_question = row["original_question"]
            person = row["person"]

            # Convert prompt to proper format if needed
            if isinstance(prompt, str):
                # If prompt is a string, assume it's JSON and parse it
                try:
                    prompt = json.loads(prompt)
                except Exception:
                    # If not JSON, create a simple user message
                    prompt = [{"role": "user", "content": prompt}]
            elif hasattr(prompt, "tolist"):
                # If it's a numpy array or pandas series, convert to list
                prompt = prompt.tolist()

            # Ensure prompt is a list of dicts
            if not isinstance(prompt, list):
                prompt = [{"role": "user", "content": str(prompt)}]

            # Generate advisor feedback
            advisor_feedback = self.generate_advisor_feedback(prompt)

            # Get student response
            student_response = self.get_student_response(
                advisor_feedback, original_question
            )

            # Compute score
            score = self.compute_reading_level_score(student_response, person)

            # Store results
            result = {
                "index": idx,
                "person": person,
                "original_question": original_question,
                "advisor_feedback": advisor_feedback,
                "student_response": student_response,
                "score": score,
                "review_length": len(student_response.split())
                if student_response
                else 0,
            }
            return result

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            # Return failed result
            return {
                "index": idx,
                "person": row["person"],
                "original_question": row["original_question"],
                "advisor_feedback": "",
                "student_response": "",
                "score": 0.0,
                "review_length": 0,
                "error": str(e),
            }

    def evaluate_dataset(
        self, dataset_path: str, max_workers: int = 12
    ) -> Dict[str, Any]:
        """Evaluate the model on a dataset using multithreading."""
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_parquet(dataset_path)

        results = []
        person_scores = {person: [] for person in LEVEL_PEOPLE}

        print(f"Evaluating {len(df)} examples with {max_workers} threads...")

        # Create list of (index, row) tuples for processing
        examples = list(df.iterrows())

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_example = {
                executor.submit(self.process_single_example, example): example[0]
                for example in examples
            }

            # Process completed tasks with progress bar
            with tqdm(total=len(examples)) as pbar:
                for future in as_completed(future_to_example):
                    result = future.result()
                    results.append(result)

                    # Update person scores
                    person = result["person"]
                    if person in person_scores:
                        person_scores[person].append(result["score"])

                    pbar.update(1)

        # Compute aggregate metrics
        all_scores = [r["score"] for r in results]
        metrics = {
            "total_examples": len(results),
            "overall_accuracy": np.mean(all_scores) if all_scores else 0.0,
            "overall_std": np.std(all_scores) if all_scores else 0.0,
            "overall_sem": np.std(all_scores) / np.sqrt(len(all_scores))
            if all_scores
            else 0.0,
            "person_accuracies": {},
            "person_counts": {},
            "avg_review_length": np.mean([r["review_length"] for r in results])
            if results
            else 0.0,
        }

        # Per-person metrics
        for person in LEVEL_PEOPLE:
            scores = person_scores[person]
            if scores:
                metrics["person_accuracies"][person] = np.mean(scores)
                metrics["person_counts"][person] = len(scores)
            else:
                metrics["person_accuracies"][person] = 0.0
                metrics["person_counts"][person] = 0

        return {"metrics": metrics, "detailed_results": results}

    def print_evaluation_report(self, evaluation_results: Dict[str, Any]):
        """Print a formatted evaluation report."""
        metrics = evaluation_results["metrics"]

        print("\n" + "=" * 60)
        print("REVIEWS LEVEL ADVISOR EVALUATION REPORT")
        print("=" * 60)

        print("\nOverall Performance:")
        print(f"  Total Examples: {metrics['total_examples']}")
        print(
            f"  Overall Accuracy: {metrics['overall_accuracy']:.3f} ± {metrics['overall_sem']:.3f} (SEM), σ = {metrics['overall_std']:.3f}"
        )
        print(f"  Average Review Length: {metrics['avg_review_length']:.1f} words")

        print("\nPer-Person Performance:")
        for person in LEVEL_PEOPLE:
            accuracy = metrics["person_accuracies"][person]
            count = metrics["person_counts"][person]
            print(f"  {person:8s}: {accuracy:.3f} ({count:3d} examples)")

        print("\nReading Level Descriptions:")
        level_descriptions = {
            "Maya": "Elementary school",
            "Rafael": "Middle school",
            "Aiden": "High school",
            "Leila": "College",
            "Jonah": "Professor",
        }
        for person in LEVEL_PEOPLE:
            print(f"  {person:8s}: {level_descriptions.get(person, 'Unknown')}")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate reviews level advisor model via an OpenAI-compatible endpoint"
    )

    # Advisor endpoint configuration
    parser.add_argument(
        "--advisor_api_base",
        type=str,
        required=True,
        help="OpenAI-compatible base URL for the advisor model, e.g., http://127.0.0.1:8000/v1",
    )
    parser.add_argument(
        "--advisor_model",
        type=str,
        default="advisor_model",
        help="Model name to send in the OpenAI-compatible request",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/reviews/validation_level.parquet",
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use as student (writing reviews based on advisor feedback)",
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = ReviewsLevelEvaluator(
        advisor_model=args.advisor_model,
        advisor_api_base=args.advisor_api_base,
        student_model=args.student_model,
    )

    # Run evaluation
    evaluation_results = evaluator.evaluate_dataset(args.dataset_path)

    # Print report
    evaluator.print_evaluation_report(evaluation_results)


if __name__ == "__main__":
    main()
