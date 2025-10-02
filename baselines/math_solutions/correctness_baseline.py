"""
Correctness baseline for GPT-4o-mini on math solutions evaluation dataset.
Tests how many problems GPT-4o-mini gets correct without any advisor guidance.

python baselines/math_solutions/correctness_baseline.py
"""

import pandas as pd
import math_verify
from openai import OpenAI
from tqdm import tqdm
import statistics
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed


def solve_math_problem(client: OpenAI, problem: str) -> str:
    """Solve a math problem using GPT-4o-mini with simple prompt."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful math tutor. Solve the given math problem step by step and provide your final answer in \\boxed{} format.",
        },
        {"role": "user", "content": f"Solve this math problem:\n\n{problem}"},
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return ""


def compute_correctness_score(solution: str, ground_truth: str) -> float:
    """Compute correctness score using math_verify (same as environment)."""
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


def process_single_example(idx_row_tuple):
    """Process a single example - designed for multithreading."""
    idx, row = idx_row_tuple
    try:
        problem = row["original_question"]
        ground_truth = row.get("ground_truth_answer")
        student = row.get("student", "unknown")

        if not ground_truth:
            print(f"Warning: No ground truth for example {idx}")
            return None

        # Create client for this thread
        client = OpenAI()

        # Get solution from GPT-4o-mini
        solution = solve_math_problem(client, problem)

        # Compute correctness
        correctness_score = compute_correctness_score(solution, ground_truth)

        return {
            "index": idx,
            "student": student,
            "problem": problem,
            "ground_truth": ground_truth,
            "solution": solution,
            "correctness_score": correctness_score,
        }
    except Exception as e:
        print(f"Error processing example {idx}: {e}")
        return {
            "index": idx,
            "student": row.get("student", "unknown"),
            "problem": row.get("original_question", ""),
            "ground_truth": row.get("ground_truth_answer", ""),
            "solution": "",
            "correctness_score": 0.0,
            "error": str(e),
        }


def evaluate_correctness_baseline(
    dataset_path: str, max_examples: int = None, max_workers: int = 12
):
    """Evaluate GPT-4o-mini correctness baseline on math solutions dataset."""
    print("Loading dataset...")
    df = pd.read_parquet(dataset_path)

    if max_examples:
        df = df.sample(n=min(max_examples, len(df)), random_state=42)
        print(f"Evaluating on {len(df)} sampled examples")
    else:
        print(f"Evaluating on full dataset: {len(df)} examples")

    results = []

    print(f"Evaluating GPT-4o-mini correctness baseline with {max_workers} workers...")

    # Create list of (index, row) tuples for processing
    examples = list(df.iterrows())

    # Use ThreadPoolExecutor for concurrent processing
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_example = {
            executor.submit(process_single_example, example): example[0]
            for example in examples
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(examples)) as pbar:
            for future in as_completed(future_to_example):
                result = future.result()
                if result is not None:  # Skip failed examples
                    results.append(result)
                pbar.update(1)

    # Calculate overall statistics
    correctness_scores = [r["correctness_score"] for r in results]
    overall_accuracy = (
        statistics.mean(correctness_scores) if correctness_scores else 0.0
    )
    accuracy_se = (
        statistics.stdev(correctness_scores) / (len(correctness_scores) ** 0.5)
        if len(correctness_scores) > 1
        else 0.0
    )

    print("\n=== GPT-4o-mini Correctness Baseline Results ===")
    print(f"Total examples: {len(results)}")
    print(f"Overall correctness: {overall_accuracy:.3f} ± {accuracy_se:.3f} (SE)")
    print(
        f"Problems solved correctly: {sum(correctness_scores)}/{len(correctness_scores)}"
    )

    # Per-student breakdown
    from collections import defaultdict

    student_results = defaultdict(list)
    for result in results:
        student_results[result["student"]].append(result["correctness_score"])

    print("\nPer-student breakdown:")
    for student, scores in student_results.items():
        if scores:
            accuracy = statistics.mean(scores)
            count = len(scores)
            print(f"  {student}: {accuracy:.3f} ({sum(scores)}/{count} correct)")

    return {
        "overall_accuracy": overall_accuracy,
        "accuracy_se": accuracy_se,
        "total_examples": len(results),
        "correct_count": sum(correctness_scores),
        "student_results": dict(student_results),
        "detailed_results": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="GPT-4o-mini correctness baseline for math solutions"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/math_solutions/validation_hint_v2.parquet",
        help="Path to validation dataset",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (for testing)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=24,
        help="Number of threads for concurrent processing",
    )

    args = parser.parse_args()

    results = evaluate_correctness_baseline(
        args.dataset_path, args.max_examples, args.max_workers
    )

    print("\nFinal Summary:")
    print(
        f"GPT-4o-mini baseline correctness: {results['overall_accuracy']:.3f} ± {results['accuracy_se']:.3f}"
    )
    print(
        f"Solved {results['correct_count']}/{results['total_examples']} problems correctly"
    )


if __name__ == "__main__":
    main()
