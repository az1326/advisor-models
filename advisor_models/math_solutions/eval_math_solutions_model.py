"""Evaluation script for math solutions advisor model via an OpenAI-compatible endpoint.

Evaluates advisor models for math solution writing with student model feedback.
Measures correctness and style scores across different student personas.

Example usage:
    python advisor_models/math_solutions/eval_math_solutions_model.py \
        --advisor_api_base http://127.0.0.1:8000/v1 \
        --advisor_model advisor_model \
        --dataset_path data/math_solutions/validation.parquet \
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
import math_verify

from config import (
    STUDENT_SYSTEM_PROMPT,
    STUDENT_INSTRUCTION,
    STUDENTS,
    STUDENT_PERSONAS,
    STYLE_JUDGE_PROMPT,
)


class MathSolutionsEvaluator:
    """Evaluator for math solutions advisor model using an OpenAI-compatible endpoint."""

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
        self.judge_model = "gpt-4.1-mini"
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

    def get_student_response(self, advisor_feedback: str, math_problem: str) -> str:
        """Get math solution from student model using advisor feedback."""
        user_context = STUDENT_INSTRUCTION.format(
            problem=math_problem,
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

    def compute_correctness_score(self, solution: str, ground_truth: str) -> float:
        """Compute correctness score using math_verify."""
        try:
            # Extract answer from solution
            extracted_answer = math_verify.parse(solution, parsing_timeout=None)
            if extracted_answer is None:
                return 0.0

            # Verify against ground truth
            is_correct = math_verify.verify(
                extracted_answer, ground_truth, timeout_seconds=None
            )

            return 1.0 if is_correct else 0.0
        except Exception as e:
            print(f"Error computing correctness score: {e}")
            return 0.0

    def compute_style_score(self, solution: str, student: str) -> float:
        """Compute style matching score using LLM-as-a-judge."""
        if student not in STUDENT_PERSONAS:
            print(f"Warning: No persona for student {student}")
            return 0.0

        judge_criteria = STUDENT_PERSONAS[student]["judge_criteria"]
        judge_prompt = STYLE_JUDGE_PROMPT.format(
            solution=solution, judge_criteria=judge_criteria
        )

        try:
            response = self.openai_client.chat.completions.create(
                model=self.judge_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a style evaluator for math solutions.",
                    },
                    {"role": "user", "content": judge_prompt},
                ],
                temperature=0.0,
            )
            response_text = response.choices[0].message.content.strip()

            # Parse judge response
            if "ACCEPT" in response_text:
                return 1.0
            elif "PARTIAL" in response_text:
                return 0.4
            elif "REJECT" in response_text:
                return 0.0
            else:
                print(f"Warning: Unexpected judge response: {response_text}")
                return 0.0
        except Exception as e:
            print(f"Error computing style score: {e}")
            return 0.0

    def process_single_example(self, idx_row_tuple):
        """Process a single example - designed for multithreading."""
        idx, row = idx_row_tuple
        try:
            # Extract data from row
            prompt = row["prompt"]
            original_question = row["original_question"]
            student = row["student"]
            ground_truth = row.get("ground_truth_answer")

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

            # Compute scores
            correctness_score = self.compute_correctness_score(
                student_response, ground_truth
            )
            style_score = self.compute_style_score(student_response, student)
            total_score = (correctness_score + style_score) / 2.0

            # Store results
            result = {
                "index": idx,
                "student": student,
                "original_question": original_question,
                "ground_truth": ground_truth,
                "advisor_feedback": advisor_feedback,
                "student_response": student_response,
                "correctness_score": correctness_score,
                "style_score": style_score,
                "total_score": total_score,
                "solution_length": len(student_response.split())
                if student_response
                else 0,
            }
            return result

        except Exception as e:
            print(f"Error processing example {idx}: {e}")
            # Return failed result
            return {
                "index": idx,
                "student": row.get("student", "unknown"),
                "original_question": row.get("original_question", ""),
                "ground_truth": row.get("ground_truth_answer"),
                "advisor_feedback": "",
                "student_response": "",
                "correctness_score": 0.0,
                "style_score": 0.0,
                "total_score": 0.0,
                "solution_length": 0,
                "error": str(e),
            }

    def evaluate_dataset(
        self, dataset_path: str, max_workers: int = 12
    ) -> Dict[str, Any]:
        """Evaluate the model on a dataset using multithreading."""
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_parquet(dataset_path)

        results = []
        student_scores = {
            student: {"correctness": [], "style": [], "total": []}
            for student in STUDENTS
        }

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

                    # Update student scores
                    student = result["student"]
                    if student in student_scores:
                        student_scores[student]["correctness"].append(
                            result["correctness_score"]
                        )
                        student_scores[student]["style"].append(result["style_score"])
                        student_scores[student]["total"].append(result["total_score"])

                    pbar.update(1)

        # Compute aggregate metrics
        all_correctness = [r["correctness_score"] for r in results]
        all_style = [r["style_score"] for r in results]
        all_total = [r["total_score"] for r in results]

        metrics = {
            "total_examples": len(results),
            "overall_correctness": np.mean(all_correctness) if all_correctness else 0.0,
            "overall_style": np.mean(all_style) if all_style else 0.0,
            "overall_total": np.mean(all_total) if all_total else 0.0,
            "correctness_std": np.std(all_correctness) if all_correctness else 0.0,
            "style_std": np.std(all_style) if all_style else 0.0,
            "total_std": np.std(all_total) if all_total else 0.0,
            "correctness_sem": np.std(all_correctness) / np.sqrt(len(all_correctness))
            if all_correctness
            else 0.0,
            "style_sem": np.std(all_style) / np.sqrt(len(all_style))
            if all_style
            else 0.0,
            "total_sem": np.std(all_total) / np.sqrt(len(all_total))
            if all_total
            else 0.0,
            "student_metrics": {},
            "student_counts": {},
            "avg_solution_length": np.mean([r["solution_length"] for r in results])
            if results
            else 0.0,
        }

        # Per-student metrics
        for student in STUDENTS:
            scores = student_scores[student]
            if scores["total"]:
                metrics["student_metrics"][student] = {
                    "correctness": np.mean(scores["correctness"]),
                    "style": np.mean(scores["style"]),
                    "total": np.mean(scores["total"]),
                }
                metrics["student_counts"][student] = len(scores["total"])
            else:
                metrics["student_metrics"][student] = {
                    "correctness": 0.0,
                    "style": 0.0,
                    "total": 0.0,
                }
                metrics["student_counts"][student] = 0

        return {"metrics": metrics, "detailed_results": results}

    def print_evaluation_report(self, evaluation_results: Dict[str, Any]):
        """Print a formatted evaluation report."""
        metrics = evaluation_results["metrics"]

        print("\n" + "=" * 70)
        print("MATH SOLUTIONS ADVISOR EVALUATION REPORT")
        print("=" * 70)

        print("\nOverall Performance:")
        print(f"  Total Examples: {metrics['total_examples']}")
        print(
            f"  Overall Correctness: {metrics['overall_correctness']:.3f} ± {metrics['correctness_sem']:.3f} (SEM), σ = {metrics['correctness_std']:.3f}"
        )
        print(
            f"  Overall Style Match: {metrics['overall_style']:.3f} ± {metrics['style_sem']:.3f} (SEM), σ = {metrics['style_std']:.3f}"
        )
        print(
            f"  Overall Total Score: {metrics['overall_total']:.3f} ± {metrics['total_sem']:.3f} (SEM), σ = {metrics['total_std']:.3f}"
        )
        print(f"  Average Solution Length: {metrics['avg_solution_length']:.1f} words")

        print("\nPer-Student Performance:")
        print(
            f"{'Student':<8} {'Count':<6} {'Correctness':<12} {'Style':<8} {'Total':<8} {'Description'}"
        )
        print("-" * 70)
        for student in STUDENTS:
            student_metrics = metrics["student_metrics"][student]
            count = metrics["student_counts"][student]
            description = STUDENT_PERSONAS[student]["style"]
            print(
                f"{student:<8} {count:<6} {student_metrics['correctness']:<12.3f} "
                f"{student_metrics['style']:<8.3f} {student_metrics['total']:<8.3f} {description}"
            )

        print("\nStudent Persona Descriptions:")
        for student in STUDENTS:
            persona = STUDENT_PERSONAS[student]
            print(f"  {student}: {persona['description']}")

        print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate math solutions advisor model via an OpenAI-compatible endpoint"
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
        default="data/math_solutions/validation.parquet",
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--student_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use as student (solving math problems based on advisor feedback)",
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = MathSolutionsEvaluator(
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
