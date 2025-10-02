"""Dataset construction for math domain.

Generates training and validation datasets for math problem solving with advisor feedback.
Uses math_verify for answer extraction and verification.

Example usage:
    python advisor_models/math/construct_dataset.py \
    --output_dir data/math \
    --model gpt-4o-mini \
    --train_size 1000 \
    --val_size 500
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, load_dataset
import litellm
from math_verify import parse, verify

from config import STUDENT_INITIAL_SYSTEM_PROMPT, ADVISOR_TEMPLATE

# Configuration constants
TRAIN_DATASET = "agentica-org/DeepScaleR-Preview-Dataset"
VALIDATION_FILE = "math500.jsonl"


def extract_answer(response_str: str):
    """Extract the final answer from a math response using math_verify.parse."""
    try:
        parsed_answer = parse(response_str, parsing_timeout=None)
        return parsed_answer
    except Exception as e:
        print(f"Error parsing answer: {e}")
        return None


def compute_score(extracted_answer, ground_truth: str) -> float:
    """Compute score by comparing extracted answer with ground truth using math_verify.verify."""
    try:
        parsed_ground_truth = parse(ground_truth, parsing_timeout=None)
        if extracted_answer is None:
            return 0.0
        is_correct = verify(parsed_ground_truth, extracted_answer, timeout_seconds=None)
        return 1.0 if is_correct else 0.0
    except Exception as e:
        print(f"Error computing score: {e}")
        return 0.0


def build_advisor_prompt(question: str, initial_response: str) -> str:
    """Build the advisor prompt for a given question and initial response."""
    return ADVISOR_TEMPLATE.format(question=question, initial_response=initial_response)


def get_initial_response(question: str, model: str = "gpt-4o-mini") -> str:
    """Get initial response from student model."""
    try:
        litellm.drop_params = True
        messages = [
            {"role": "system", "content": STUDENT_INITIAL_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        response = litellm.completion(
            model=model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting initial response: {e}")
        return ""


def load_train_data() -> List[Dict[str, str]]:
    """Load training data from HuggingFace DeepScaleR dataset."""
    print(f"Loading training data from {TRAIN_DATASET}...")

    try:
        dataset = load_dataset(TRAIN_DATASET, split="train")
        questions = []

        for item in dataset:
            if "problem" in item and "answer" in item:
                questions.append(
                    {"question": item["problem"], "answer": str(item["answer"])}
                )

        print(f"Loaded {len(questions)} training questions")
        return questions

    except Exception as e:
        print(f"Error loading training data: {e}")
        return []


def load_validation_data(data_dir: Path) -> List[Dict[str, str]]:
    """Load validation data from local math500.jsonl file."""
    val_file = data_dir / VALIDATION_FILE

    if not val_file.exists():
        print(f"Warning: Validation file {val_file} not found")
        return []

    questions = []
    with open(val_file, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            if "question" in item and "answer" in item:
                questions.append(
                    {"question": item["question"], "answer": str(item["answer"])}
                )
            elif "problem" in item and "answer" in item:
                questions.append(
                    {"question": item["problem"], "answer": str(item["answer"])}
                )

    print(f"Loaded {len(questions)} validation questions")
    return questions


def process_question(args_tuple) -> Dict[str, Any]:
    """Process a single question to create dataset row."""
    question_data, model = args_tuple
    question = question_data["question"]
    ground_truth = question_data["answer"]

    # Get initial response from student model
    initial_response = get_initial_response(question, model)

    # Build advisor prompt
    advisor_prompt = build_advisor_prompt(question, initial_response)

    # Compute initial reward for tracking
    extracted_initial = extract_answer(initial_response)
    initial_reward = compute_score(extracted_initial, ground_truth)

    # Create the prompt structure for training
    prompt = [{"role": "user", "content": advisor_prompt}]

    return {
        "prompt": prompt,
        "env_class": "math",
        "reward_spec": {"ground_truth": ground_truth},
        "original_question": question,
        "original_response": initial_response,
        "model": model,
        "initial_reward": initial_reward,
    }


def process_questions(
    questions: List[Dict[str, str]], model: str, max_workers: int = 10
) -> List[Dict[str, Any]]:
    """Process multiple questions in parallel."""
    print(f"Processing {len(questions)} questions with {model}...")

    # Prepare arguments for parallel processing
    args_list = [(q, model) for q in questions]

    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_args = {
            executor.submit(process_question, args): args for args in args_list
        }

        # Collect results with progress bar
        for future in tqdm(
            as_completed(future_to_args),
            total=len(args_list),
            desc="Processing questions",
        ):
            try:
                result = future.result()
                rows.append(result)
            except Exception as e:
                print(f"Error processing question: {e}")

    return rows


def main():
    parser = argparse.ArgumentParser(description="Construct math domain dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for dataset files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for initial responses",
    )
    parser.add_argument(
        "--max_workers", type=int, default=12, help="Number of parallel workers"
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=None,
        help="Limit training set size (for testing)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=None,
        help="Limit validation set size (for testing)",
    )

    args = parser.parse_args()
    random.seed(42)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load datasets
    current_dir = Path(__file__).parent
    train_questions = load_train_data()
    val_questions = load_validation_data(current_dir)

    # Limit sizes if specified
    if args.train_size and len(train_questions) > args.train_size:
        train_questions = random.sample(train_questions, args.train_size)
    if args.val_size and len(val_questions) > args.val_size:
        val_questions = random.sample(val_questions, args.val_size)

    # Process questions
    train_rows = process_questions(train_questions, args.model, args.max_workers)
    val_rows = process_questions(val_questions, args.model, args.max_workers)

    # Save datasets
    if train_rows:
        filename = f"train_{args.model}_{args.train_size}.parquet"
        Dataset.from_pandas(pd.DataFrame(train_rows)).to_parquet(output_dir / filename)
        print(f"Saved {len(train_rows)} training examples")
        print(
            f"Training initial accuracy: {sum(row['initial_reward'] for row in train_rows) / len(train_rows):.3f}"
        )

    if val_rows:
        filename = f"validation_{args.model}_{args.val_size}.parquet"
        Dataset.from_pandas(pd.DataFrame(val_rows)).to_parquet(output_dir / filename)
        print(f"Saved {len(val_rows)} validation examples")
        print(
            f"Validation initial accuracy: {sum(row['initial_reward'] for row in val_rows) / len(val_rows):.3f}"
        )


if __name__ == "__main__":
    main()
