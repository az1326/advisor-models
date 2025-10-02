"""Dataset construction for math solutions domain.

Generates training and validation datasets for math solution writing with advisor feedback.
Uses Math500 dataset as the source for mathematical problems.

Example usage:
    python advisor_models/math_solutions/construct_dataset.py \
        --output_dir data/math_solutions \
        --model gpt-4o-mini \
        --train_ratio 0.8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from tqdm import tqdm

import datasets

from config import (
    ADVISOR_SYSTEM_PROMPT,
    ADVISOR_INSTRUCTION,
    STUDENTS,
    STUDENT_PERSONAS,
)


def build_advisor_prompt(task: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build the advisor prompt that the model will receive."""
    user_content = ADVISOR_INSTRUCTION.format(
        student=task["student"],
        problem=task["problem"],
    )

    return [
        {"role": "system", "content": ADVISOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def load_math500_data() -> List[Dict[str, Any]]:
    """Load Math500 dataset from JSONL file."""
    current_dir = Path(__file__).parent
    math500_path = current_dir / "math500.jsonl"

    if not os.path.exists(math500_path):
        raise FileNotFoundError(f"Math500 dataset not found at {math500_path}")

    problems = []
    with open(math500_path, "r") as f:
        for line in f:
            problems.append(json.loads(line.strip()))

    print(f"Loaded {len(problems)} problems from Math500 dataset")
    return problems


def split_math500_data(
    problems: List[Dict[str, Any]], train_ratio: float = 0.8
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split Math500 data into train and validation sets."""
    # Shuffle the problems
    shuffled_problems = problems.copy()
    random.shuffle(shuffled_problems)

    # Split into train/val
    split_idx = int(len(shuffled_problems) * train_ratio)
    train_problems = shuffled_problems[:split_idx]
    val_problems = shuffled_problems[split_idx:]

    print(
        f"Split into {len(train_problems)} train and {len(val_problems)} validation problems"
    )
    return train_problems, val_problems


def generate_math_tasks(problems: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate math solution tasks by assigning problems to different students."""
    tasks = []

    for i, problem_data in enumerate(problems):
        # Cycle through students for each problem
        student = STUDENTS[i % len(STUDENTS)]

        task = {
            "problem": problem_data["problem"],
            "answer": problem_data["answer"],
            "student": student,
            "judge_criteria": STUDENT_PERSONAS[student]["judge_criteria"],
        }
        tasks.append(task)

    return tasks


def process_math_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single math task to create a training example."""
    # Build the advisor prompt
    prompt = build_advisor_prompt(task)

    return {
        "prompt": prompt,
        "env_class": "math_solutions",
        "reward_spec": {
            "judge_criteria": task["judge_criteria"],
        },
        # The following keys become ``extras`` in the env
        "original_question": task["problem"],
        "student": task["student"],
        "ground_truth_answer": task["answer"],
    }


def process_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process math tasks in parallel to create training examples."""
    # Process tasks in parallel
    rows = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for row in tqdm(
            executor.map(process_math_task, tasks), desc="Processing tasks"
        ):
            rows.append(row)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct math solutions dataset")
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to output files",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of data to use for training (rest goes to validation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/math_solutions",
        help="Output directory for dataset",
    )

    args = parser.parse_args()
    random.seed(42)

    print("Loading Math500 dataset...")
    problems = load_math500_data()

    print("Splitting data into train/validation...")
    train_problems, val_problems = split_math500_data(problems, args.train_ratio)

    print("Generating math solution tasks...")
    print(f"Train problems: {len(train_problems)}, Val problems: {len(val_problems)}")
    print(f"Students: {STUDENTS}")

    # Generate math solution tasks
    train_tasks = generate_math_tasks(train_problems)
    val_tasks = generate_math_tasks(val_problems)

    print(
        f"Processing {len(train_tasks)} training and {len(val_tasks)} validation tasks..."
    )

    # Process tasks to create training examples
    train_rows = process_tasks(train_tasks, "train")
    val_rows = process_tasks(val_tasks, "validation")

    # Write to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.suffix}" if args.suffix else ""
    train_parquet_path = os.path.join(args.output_dir, f"train{suffix}.parquet")
    val_parquet_path = os.path.join(args.output_dir, f"validation{suffix}.parquet")

    datasets.Dataset.from_list(train_rows).to_parquet(train_parquet_path)
    datasets.Dataset.from_list(val_rows).to_parquet(val_parquet_path)

    print(
        f"Wrote {len(train_rows)} training and {len(val_rows)} validation examples to {args.output_dir}"
    )
