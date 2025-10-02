"""Dataset construction for MTOB (Machine Translation from One Book) domain.

Generates training and validation datasets for machine translation with advisor feedback.
Uses the MTOB dataset from the official repository for Kalamang->English translation.

Example usage:
    python advisor_models/mtob/construct_dataset.py \
        --output_dir data/mtob \
        --train_size 200 \
        --val_size 50

The script writes ``train.parquet`` and ``validation.parquet`` to ``output_dir``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import pylcs
from tqdm import tqdm

from config import ADVISOR_INITIAL_INSTRUCTIONS, ADVISOR_FINAL_INSTRUCTIONS


def load_mtob_train_data(mtob_path: str) -> List[Dict[str, Any]]:
    """Load MTOB training examples.

    Args:
        mtob_path: Path to the MTOB official repository

    Returns:
        List of examples with 'kalamang' and 'english' keys
    """
    splits_dir = Path(mtob_path) / "splits"
    train_file = splits_dir / "train_examples.json"

    if not train_file.exists():
        raise FileNotFoundError(f"MTOB train file not found: {train_file}")

    with open(train_file, "r", encoding="utf-8") as f:
        data = json.load(f)[1:]

    examples = []
    for item in data:
        examples.append({"source": item["translation"], "target": item["original"]})

    return examples


def load_mtob_test_data(mtob_path: str) -> List[Dict[str, Any]]:
    """Load MTOB test examples.

    Args:
        mtob_path: Path to the MTOB official repository

    Returns:
        List of examples with 'source' and 'target' keys
    """
    splits_dir = Path(mtob_path) / "splits"

    test_file = splits_dir / "test_examples_ke.json"

    if not test_file.exists():
        raise FileNotFoundError(f"MTOB test file not found: {test_file}")

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)[1:]

    examples = []
    for item in data:
        examples.append({"source": item["original"], "target": item["ground_truth"]})
    return examples


def load_wordlist(wordlist_path: str) -> Dict[str, Any]:
    """Load MTOB wordlist."""
    with open(wordlist_path, "r") as f:
        return json.load(f)


def get_wordlist_references_lcs(
    source_text: str, wordlist_dict: Dict, num_words: int = 2
) -> List[str]:
    """Get wordlist references using LCS matching from MTOB baseline."""
    references = []

    # Extract words from source text
    input_words = source_text.strip().replace(".", "").replace(",", "").split(" ")

    # Kalamang to English
    examples = [
        {"kalamang": k, "pos": p, "english": e}
        for k, (p, e) in wordlist_dict["ke"].items()
    ]
    examples_with_space = [f" {ex['kalamang']} " for ex in examples]

    for input_word in input_words:
        if not input_word.strip():
            continue
        longest_common_substring_lengths = np.array(
            pylcs.lcs_string_of_list(f" {input_word} ", examples_with_space)
        )
        indices = np.argpartition(longest_common_substring_lengths, -num_words)[
            -num_words:
        ]

        for index in indices:
            example = examples[index.item()]
            ref_text = f'A lexically close dictionary entry for "{input_word}": {example["kalamang"]} ({example["pos"]}) = {example["english"]}'
            references.append(ref_text)

    return references


def get_sentence_references_lcs(
    source_text: str, train_examples: List[Dict], num_sentences: int = 2
) -> List[str]:
    """Get sentence references using LCS matching from MTOB baseline."""
    references = []

    # Extract words from source text
    input_words = source_text.strip().replace(".", "").replace(",", "").split(" ")

    # Clean source text for exact match comparison
    clean_source = source_text.strip().replace(".", "").replace(",", "")

    # Kalamang to English
    examples = [
        {"kalamang": d["source"], "english": d["target"]} for d in train_examples
    ]
    examples_with_space = [f" {ex['kalamang']} " for ex in examples]

    for input_word in input_words:
        if not input_word.strip():
            continue
        longest_common_substring_lengths = np.array(
            pylcs.lcs_string_of_list(f" {input_word} ", examples_with_space)
        )

        # Get more indices than needed to account for potential exact matches
        max_indices = min(len(examples), num_sentences + 1)
        indices = np.argpartition(longest_common_substring_lengths, -max_indices)[
            -max_indices:
        ]
        # Sort by LCS score (highest first)
        indices = indices[np.argsort(longest_common_substring_lengths[indices])[::-1]]

        added_count = 0
        for index in indices:
            if added_count >= num_sentences:
                break
            example = examples[index.item()]
            # Skip if this is the exact same sentence as the source
            clean_example = (
                example["kalamang"].strip().replace(".", "").replace(",", "")
            )
            if clean_example == clean_source:
                continue
            ref_text = f'A reference sentence for "{input_word}": "{example["kalamang"]}" → "{example["english"]}"'
            references.append(ref_text)
            added_count += 1

    return references


def build_advisor_prompt(
    source_text: str,
    wordlist_dict: Optional[Dict] = None,
    train_examples: Optional[List[Dict]] = None,
    num_reference_words: int = 2,
    num_reference_sentences: int = 2,
) -> str:
    """Build advisor prompt with optional reference materials using MTOB baseline approach."""

    # Start with base translation instruction
    base_prompt = ADVISOR_INITIAL_INSTRUCTIONS.format(source_text=source_text)

    reference_sections = []

    # Add wordlist references using LCS
    wordlist_refs = get_wordlist_references_lcs(
        source_text, wordlist_dict, num_reference_words
    )
    reference_sections.extend(wordlist_refs)

    # Add sentence references using LCS
    sentence_refs = get_sentence_references_lcs(
        source_text, train_examples, num_reference_sentences
    )
    reference_sections.extend(sentence_refs)

    # Combine all sections
    if reference_sections:
        full_prompt = (
            base_prompt + "\n\nReference materials:\n" + "\n".join(reference_sections)
        )
    else:
        full_prompt = base_prompt

    # Add final instruction
    full_prompt += ADVISOR_FINAL_INSTRUCTIONS.format(source_text=source_text)

    return full_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Generate MTOB dataset for RL training"
    )
    parser.add_argument(
        "--mtob_path",
        type=str,
        default="advisor_models/mtob/mtob-official",
        help="Path to MTOB official repository",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=100,
        help="Number of training examples (sampled from train set)",
    )
    parser.add_argument(
        "--val_size",
        type=int,
        default=50,
        help="Number of validation examples (sampled from test set)",
    )
    parser.add_argument(
        "--num_reference_words",
        type=int,
        default=2,
        help="Number of words to use from wordlist",
    )
    parser.add_argument(
        "--num_reference_sentences",
        type=int,
        default=2,
        help="Number of sentences to use from train examples",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to output file names",
    )

    args = parser.parse_args()
    random.seed(42)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading MTOB data from {args.mtob_path}")
    try:
        # Load train data for training
        full_train_examples = load_mtob_train_data(args.mtob_path)
        print(f"Loaded {len(full_train_examples)} training examples")

        # Load test data for validation
        val_examples = load_mtob_test_data(args.mtob_path)
        print(f"Loaded {len(val_examples)} test examples for validation")

        # Load reference materials if needed
        wordlist_dict = load_wordlist(f"{args.mtob_path}/resources/wordlist.json")
        print("Loaded wordlist for reference materials")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the MTOB dataset is properly extracted and accessible.")
        return

    # Sample from train and test sets if requested sizes are smaller
    if len(full_train_examples) > args.train_size:
        train_examples = random.sample(full_train_examples, args.train_size)
        print(f"Sampled {args.train_size} training examples")

    if len(val_examples) > args.val_size:
        val_examples = random.sample(val_examples, args.val_size)
        print(f"Sampled {args.val_size} validation examples")

    def process_examples(examples_list: List[Dict]) -> List[Dict]:
        """Process examples to create training data."""
        processed = []

        for example in tqdm(examples_list, desc="Processing examples"):
            source_text = example["source"]
            ground_truth = example["target"]

            # Build advisor prompt with optional reference materials
            advisor_prompt = build_advisor_prompt(
                source_text,
                wordlist_dict=wordlist_dict,
                train_examples=full_train_examples,
                num_reference_words=args.num_reference_words,
                num_reference_sentences=args.num_reference_sentences,
            )

            # Pre-build reference materials for student model
            reference_materials = []
            wordlist_refs = get_wordlist_references_lcs(
                source_text, wordlist_dict, args.num_reference_words
            )
            reference_materials.extend(wordlist_refs)

            sentence_refs = get_sentence_references_lcs(
                source_text,
                full_train_examples,
                args.num_reference_sentences,
            )
            reference_materials.extend(sentence_refs)

            assert len(reference_materials) > 0, "No reference materials found"
            # Create training example with pre-built reference materials
            row = {
                "prompt": [{"role": "user", "content": advisor_prompt}],
                "env_class": "mtob",
                "original_question": source_text,
                "reward_spec": {"ground_truth": ground_truth},
                "reference_materials": reference_materials,
            }

            processed.append(row)

        return processed

    # Process train and validation sets
    print("Processing training examples...")
    train_data = process_examples(train_examples)

    print("Processing validation examples...")
    val_data = process_examples(val_examples)

    # Save to parquet files
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    if args.suffix is None or args.suffix == "":
        suffix = ""
    else:
        suffix = "_" + args.suffix

    train_path = Path(args.output_dir) / f"train{suffix}.parquet"
    val_path = Path(args.output_dir) / f"validation{suffix}.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)

    print(f"Saved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(val_data)} validation examples to {val_path}")


if __name__ == "__main__":
    main()
