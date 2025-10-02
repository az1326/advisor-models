"""Dataset construction for reviews length domain.

Generates training and validation datasets for review writing with length preferences.
Uses unique prompts from HuggingFace dataset for diverse review generation tasks.

Example usage:
    python advisor_models/reviews/construct_dataset_length.py \
        --output_dir data/reviews
"""

from __future__ import annotations

import argparse
import os
import random
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any
from tqdm import tqdm

import datasets
from datasets import load_dataset

from config import (
    LENGTH_ADVISOR_SYSTEM_PROMPT,
    WEAK_ADVISOR_SYSTEM_PROMPT,
    LENGTH_ADVISOR_INSTRUCTION,
    WEAK_ADVISOR_INSTRUCTION,
    LENGTH_PEOPLE,
    LENGTH_CRITERIA,
)


def build_advisor_prompt(
    task: Dict[str, Any], use_weak_prompt: bool = False
) -> List[Dict[str, str]]:
    """Build the advisor prompt that the model will receive."""
    if use_weak_prompt:
        system_prompt = WEAK_ADVISOR_SYSTEM_PROMPT
        instruction_template = WEAK_ADVISOR_INSTRUCTION
    else:
        system_prompt = LENGTH_ADVISOR_SYSTEM_PROMPT
        instruction_template = LENGTH_ADVISOR_INSTRUCTION

    user_content = instruction_template.format(
        person=task["person"],
        prompt=task["prompt"],
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def load_unique_prompts() -> tuple[List[str], List[str]]:
    """Load unique prompts from the HuggingFace dataset."""
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("Asap7772/steered_reviews_full_autolabel")

    # Get unique prompts from train and test sets
    train_prompts = list(set(dataset["train"]["prompt"]))
    test_prompts = list(set(dataset["test"]["prompt"]))

    print(f"Found {len(train_prompts)} unique prompts in train set")
    print(f"Found {len(test_prompts)} unique prompts in test set")

    return train_prompts, test_prompts


def generate_review_tasks(prompts: List[str]) -> List[Dict[str, Any]]:
    """Generate review tasks by assigning prompts to different people."""
    tasks = []

    for i, prompt in enumerate(prompts):
        # Cycle through people for each prompt
        person = LENGTH_PEOPLE[i % len(LENGTH_PEOPLE)]

        task = {
            "prompt": prompt,
            "person": person,
            "length_preference": LENGTH_CRITERIA[person],
        }

        tasks.append(task)

    return tasks


def process_review_task(args_tuple) -> Dict[str, Any]:
    """Process a single review task to create a training example."""
    task, use_weak_prompt = args_tuple

    # Build the advisor prompt
    prompt = build_advisor_prompt(task, use_weak_prompt)

    return {
        "prompt": prompt,
        "env_class": "reviews_length",
        "reward_spec": {
            "length": task["length_preference"],
        },
        # The following keys become ``extras`` in the env
        "original_question": task["prompt"],
        "person": task["person"],
    }


def process_tasks(
    tasks: List[Dict[str, Any]], use_weak_prompt: bool = False
) -> List[Dict[str, Any]]:
    """Process review tasks in parallel to create training examples."""
    # Prepare arguments for parallel processing
    process_args = [(task, use_weak_prompt) for task in tasks]

    # Process tasks in parallel
    rows = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for row in tqdm(
            executor.map(process_review_task, process_args),
            desc="Processing tasks",
        ):
            rows.append(row)

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct reviews dataset")
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to add to output files",
    )
    parser.add_argument(
        "--use_weak_prompt",
        action="store_true",
        help="Use weak prompt for advisor prompt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/reviews",
        help="Output directory for dataset",
    )

    args = parser.parse_args()
    random.seed(42)

    print("Loading unique prompts from HuggingFace dataset...")
    train_prompts, test_prompts = load_unique_prompts()

    print("Generating review tasks...")
    print(f"Train prompts: {len(train_prompts)}, Test prompts: {len(test_prompts)}")
    print(f"People: {LENGTH_PEOPLE}")

    # Generate review tasks
    train_tasks = generate_review_tasks(train_prompts)
    val_tasks = generate_review_tasks(test_prompts)

    print(
        f"Processing {len(train_tasks)} training and {len(val_tasks)} validation tasks..."
    )

    # Process tasks to create training examples
    train_rows = process_tasks(train_tasks, args.use_weak_prompt)
    val_rows = process_tasks(val_tasks, args.use_weak_prompt)

    # Write to parquet
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = f"_{args.suffix}" if args.suffix else ""
    train_parquet_path = os.path.join(args.output_dir, f"train_length{suffix}.parquet")
    val_parquet_path = os.path.join(
        args.output_dir, f"validation_length{suffix}.parquet"
    )

    datasets.Dataset.from_list(train_rows).to_parquet(train_parquet_path)
    datasets.Dataset.from_list(val_rows).to_parquet(val_parquet_path)

    print(
        f"Wrote {len(train_rows)} training and {len(val_rows)} validation examples to {args.output_dir}"
    )
