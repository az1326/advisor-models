"""Dataset construction for rule arena (US tax) domain.

Generates training and validation datasets for US tax calculation with advisor feedback.
Creates datasets with tax forms and calculation problems for advisor model training.

Example usage:
    python advisor_models/rule_arena/construct_dataset.py \
        --output_dir data/rule_arena \
        --train_size 75 \
        --val_size 25
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datasets import Dataset
import litellm
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from config import PROBLEM_TEMPLATE, ADVISOR_INSTRUCTIONS

### Scoring Functions
import re

# Add RuleArena to path
sys.path.append(str(Path(__file__).parent / "RuleArena"))


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


def compute_score(extracted_answer: str, ground_truth: str) -> float:
    """Compute the score."""
    if not extracted_answer:
        return 0.0

    # For tax amounts, use numpy.isclose like RuleArena reference
    try:
        extracted_val = float(extracted_answer.replace(",", "").replace("$", ""))
        ground_truth_val = float(str(ground_truth).replace(",", "").replace("$", ""))
        # Use numpy.isclose for comparison (matches RuleArena implementation)
        return 1.0 if np.isclose(extracted_val, ground_truth_val) else 0.0
    except (ValueError, TypeError):
        return 0.0


def get_initial_response(question: str, model: str = "gpt-4.1-mini") -> str:
    """Query *model* for an initial attempt using LiteLLM. Returns an empty string on fail."""
    try:
        litellm.drop_params = True
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": question}],
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error getting initial response: {e}")
        return ""


def build_advisor_prompt(question: str, initial_response: str) -> List[Dict[str, str]]:
    """Compose the advisor prompt."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": initial_response},
        {"role": "user", "content": ADVISOR_INSTRUCTIONS},
    ]


def load_synthesized_problems(data_path: str) -> List[Dict[str, Any]]:
    """Load synthesized problems from JSON/JSONL files."""
    problems_file = Path(data_path)

    if not problems_file.exists():
        raise FileNotFoundError(f"Synthesized problems file not found: {problems_file}")

    problems = []

    with open(problems_file, "r") as f:
        data_list = json.load(f)

    # Tax files - build prompts from taxpayer data
    for data in data_list:
        tax_payer_pydantic = data.get("pydantic", {})
        tax_payer_dict = data.get("dict", {})

        # Store both structures - pydantic for TaxPayer creation, dict for forms
        combined_info = {
            "pydantic": tax_payer_pydantic,
            "dict": tax_payer_dict,
        }

        problems.append(
            {
                "info_dict": combined_info,
            }
        )

    return problems


def compute_ground_truth(info_dict: Dict[str, Any]) -> int:
    """Compute ground truth answer from info_dict."""
    # Import the compute_answer function from RuleArena tax
    sys.path.append(str(Path(__file__).parent / "RuleArena" / "tax"))
    try:
        from micro_evaluation import compute_answer
        from structured_forms import TaxPayer
    except ImportError as e:
        raise ImportError(
            f"Failed to import required tax modules: {e}. "
            f"Make sure RuleArena tax files are available."
        )

    # Convert the info_dict to a TaxPayer object and compute the ground truth
    # Use the pydantic structure for TaxPayer creation
    try:
        # Extract the pydantic data for TaxPayer creation
        pydantic_data = info_dict.get("pydantic", info_dict)
        tax_payer = TaxPayer(**pydantic_data)
        # compute_answer returns a tuple (answer, _), we need the first element
        ground_truth, _ = compute_answer(tax_payer)
        return ground_truth
    except Exception as e:
        raise Exception(f"Error computing tax ground truth: {e}")


def build_student_prompt(tax_payer_dict: Dict[str, Any] = None) -> str:
    """Build the initial student prompt."""
    # Import tax forms and example from RuleArena tax
    sys.path.append(str(Path(__file__).parent / "RuleArena" / "tax"))

    try:
        from prompt import (
            basic_forms,
            itemized_forms,
            self_employ_forms,
            edu_forms,
            schedule_8812,
        )
    except ImportError as e:
        raise ImportError(
            f"Failed to import required tax modules: {e}. "
            f"Make sure RuleArena tax files are available."
        )

    # Build forms based on taxpayer characteristics (like auto_test.py)
    forms = [basic_forms]

    # Extract the right data structures
    pydantic_data = tax_payer_dict.get("pydantic", {}) if tax_payer_dict else {}
    dict_data = tax_payer_dict.get("dict", {}) if tax_payer_dict else {}

    # Use pydantic data for taxpayer characteristics
    taxpayer_info = pydantic_data if pydantic_data else tax_payer_dict
    if taxpayer_info:
        if taxpayer_info.get("itemized", False):
            forms.append(itemized_forms)
        if taxpayer_info.get("self_employed", False):
            forms.append(self_employ_forms)
        if taxpayer_info.get("has_student_loans_or_education_expenses", False):
            forms.append(edu_forms)
        if taxpayer_info.get("child_and_dependent", False):
            forms.append(schedule_8812)

    forms_str = "".join(forms)

    # Replace data fields in forms using dict data (which has the 'data' field)
    if dict_data and "data" in dict_data:
        for k, v in dict_data["data"].items():
            forms_str = forms_str.replace(
                "$" + k, "$" + f"{v:,}" if not isinstance(v, str) else v
            )

    # Build the full prompt
    full_prompt = PROBLEM_TEMPLATE.replace("$forms", forms_str)

    # Replace taxpayer personal info if available
    if taxpayer_info:
        full_prompt = full_prompt.replace("$name", taxpayer_info.get("name", "Unknown"))
        full_prompt = full_prompt.replace("$age", str(taxpayer_info.get("age", "")))
        full_prompt = full_prompt.replace(
            "$spouse_age", str(taxpayer_info.get("spouse_age", ""))
        )
        full_prompt = full_prompt.replace(
            "$blind", str(taxpayer_info.get("blind", False))
        )
        full_prompt = full_prompt.replace(
            "$spouse_blind", str(taxpayer_info.get("spouse_blind", False))
        )
        full_prompt = full_prompt.replace(
            "$filing_status", taxpayer_info.get("filing_status", "Unknown")
        )
        full_prompt = full_prompt.replace(
            "$itemized", str(taxpayer_info.get("itemized", False))
        )
        full_prompt = full_prompt.replace(
            "$num_qualifying_children",
            str(taxpayer_info.get("num_qualifying_children", 0)),
        )
        full_prompt = full_prompt.replace(
            "$num_other_dependents", str(taxpayer_info.get("num_other_dependents", 0))
        )

    return full_prompt


def get_problems_from_data_path(
    data_path: str, num_problems: int
) -> List[Dict[str, Any]]:
    """Get problems from a specific data path."""
    # Load synthesized problems
    all_problems = load_synthesized_problems(data_path)

    # Shuffle and take the requested number
    random.shuffle(all_problems)
    selected_problems = all_problems[:num_problems]

    # Compute ground truth for each problem
    problems_with_gt = []
    for problem in selected_problems:
        try:
            ground_truth = compute_ground_truth(problem["info_dict"])

            problems_with_gt.append(
                {
                    "ground_truth": ground_truth,
                    "info_dict": problem["info_dict"],
                }
            )
        except Exception as e:
            print(f"Error computing ground truth for problem: {e}")
            continue

    return problems_with_gt


def process_problem(
    problem: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    """Process a single problem with OpenAI API call."""
    ground_truth = problem["ground_truth"]
    tax_payer_dict = problem.get("info_dict", {})
    full_question = build_student_prompt(tax_payer_dict)

    # Get initial response using the full prompt
    initial_response = get_initial_response(full_question, model)

    # Compute initial reward using the same scoring function as the environment
    extracted_answer = extract_answer(initial_response)
    initial_reward = compute_score(extracted_answer, ground_truth)

    # Build advisor prompt - the advisor will see the original question and initial response
    # and provide feedback to improve the response
    advisor_prompt = [
        {"role": "user", "content": full_question},
        {"role": "assistant", "content": initial_response},
        {
            "role": "user",
            "content": "Please review the solution and provide feedback to improve it if needed. Focus on accuracy, completeness, and following the rules correctly.",
        },
    ]

    return {
        "prompt": advisor_prompt,
        "env_class": "rule_arena",
        "reward_spec": {"ground_truth": ground_truth},
        "model": model,
        "original_question": full_question,
        "original_response": initial_response,
        "initial_reward": initial_reward,
        "info_dict": problem.get("info_dict", {}),
    }


def process_problems_parallel(
    problems: List[Dict[str, Any]],
    model: str,
    max_workers: int = 8,
) -> List[Dict[str, Any]]:
    """Process multiple problems in parallel using ThreadPoolExecutor."""
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            executor.submit(process_problem, problem, model) for problem in problems
        ]

        # Collect results as they complete
        for future in tqdm(
            as_completed(futures),
            total=len(problems),
            desc="Processing problems",
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing problem: {e}")
                continue

    return results


def main():
    parser = argparse.ArgumentParser(description="Generate RuleArena dataset")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--model", type=str, default="gpt-4.1-mini", help="Model for initial responses"
    )
    parser.add_argument(
        "--complexity", type=int, default=0, choices=[0, 1, 2], help="Complexity level"
    )
    parser.add_argument(
        "--train_size", type=int, default=75, help="Number of training examples"
    )
    parser.add_argument(
        "--val_size", type=int, default=25, help="Number of validation examples"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=12,
        help="Maximum number of parallel workers for processing",
    )

    args = parser.parse_args()
    random.seed(42)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build data path
    data_path = str(
        Path(__file__).parent
        / "RuleArena"
        / "tax"
        / "synthesized_problems"
        / f"comp_{args.complexity}.json"
    )
    print(f"Generating tax problems with complexity {args.complexity}")

    print(f"Train size: {args.train_size}, Val size: {args.val_size}")

    # Generate problems using synthesized problems
    total_problems = args.train_size + args.val_size
    problems = get_problems_from_data_path(data_path, total_problems)

    if len(problems) < total_problems:
        print(
            f"Warning: Only loaded {len(problems)} problems, requested {total_problems}"
        )
        if len(problems) == 0:
            print(f"No problems found for complexity {args.complexity}")
            return

    # Shuffle and split
    random.shuffle(problems)
    train_problems = problems[: args.train_size]
    val_problems = problems[args.train_size : args.train_size + args.val_size]

    # Process problems in parallel
    print(f"Processing training problems with {args.max_workers} workers...")
    train_rows = process_problems_parallel(
        train_problems,
        args.model,
        max_workers=args.max_workers,
    )

    print(f"Processing validation problems with {args.max_workers} workers...")
    val_rows = process_problems_parallel(
        val_problems,
        args.model,
        max_workers=args.max_workers,
    )

    # Save to parquet
    if train_rows:
        train_dataset = Dataset.from_list(train_rows)
        train_dataset.to_parquet(
            output_dir / f"train_{args.model}_{args.complexity}.parquet"
        )
        print(
            f"Saved {len(train_rows)} training examples to {output_dir / f'train_{args.model}_{args.complexity}.parquet'}"
        )

    if val_rows:
        val_dataset = Dataset.from_list(val_rows)
        val_dataset.to_parquet(
            output_dir / f"validation_{args.model}_{args.complexity}.parquet"
        )
        print(
            f"Saved {len(val_rows)} validation examples to {output_dir / f'validation_{args.model}_{args.complexity}.parquet'}"
        )

    print("Dataset generation complete!")

    # Print average initial rewards
    if train_rows:
        avg_initial_reward = sum(row["initial_reward"] for row in train_rows) / len(
            train_rows
        )
        print(f"Average initial reward for training: {avg_initial_reward}")

    if val_rows:
        avg_initial_reward = sum(row["initial_reward"] for row in val_rows) / len(
            val_rows
        )
        print(f"Average initial reward for validation: {avg_initial_reward}")


if __name__ == "__main__":
    main()
