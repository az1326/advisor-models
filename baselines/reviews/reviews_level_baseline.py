"""
GEPA baseline for reviews level domain

python baselines/reviews/reviews_level_baseline.py
"""

import dspy
from dspy.evaluate import Evaluate
from dspy import GEPA
import pandas as pd
import random
import os
import statistics
from openai import OpenAI

random.seed(42)
llm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=1.0)
dspy.settings.configure(lm=llm)

prompt_prefix = """You are a review writer. Consider the reading level preferences and style that would work best for the target person."""


def load_reviews_reading_level_data():
    """Load reviews reading level data from parquet files."""
    train_path = "data/reviews/train_level.parquet"
    val_path = "data/reviews/validation_level.parquet"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Reviews reading level data files not found at {train_path} or {val_path}"
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Convert to DSPy format
    trainset = []
    for _, row in train_df.iterrows():
        person = row["person"]
        prompt = row["original_question"]
        criteria = row["reward_spec"]["level_criteria"]
        example = dspy.Example(
            prompt=prompt_prefix + f" You are writing a review for {person}." + prompt,
            criteria=criteria,
        ).with_inputs("prompt", "criteria")
        trainset.append(example)

    valset = []
    for _, row in val_df.iterrows():
        person = row["person"]
        prompt = row["original_question"]
        criteria = row["reward_spec"]["level_criteria"]
        example = dspy.Example(
            prompt=prompt_prefix + f" You are writing a review for {person}." + prompt,
            criteria=criteria,
        ).with_inputs("prompt", "criteria")
        valset.append(example)

    return trainset, valset


def compute_reading_level_reward(review_text: str, criteria: str) -> float:
    """Compute reward based on reading level appropriateness."""
    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a reading level evaluator."},
                {"role": "user", "content": criteria.format(review=review_text)},
            ],
            temperature=0.0,
        )
        response = response.choices[0].message.content
        if "Yes" in response and "No" not in response:
            return 1.0
        return 0.0
    except Exception as e:
        print(f"Error computing review score: {e}")
        return 0.0


class ReviewGenerator(dspy.Signature):
    """Generate a review based on the given prompt."""

    prompt = dspy.InputField(desc="The review writing prompt")
    review = dspy.OutputField(desc="The generated review")


class ReviewModule(dspy.Module):
    """Review generation module."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(ReviewGenerator)

    def forward(self, prompt, criteria=None):
        return self.generate(prompt=prompt)


def compute_score_metric(example, pred, trace=None):
    """Compute the reading level reward score for a prediction."""
    review_text = pred.review
    criteria = example.criteria
    reward = compute_reading_level_reward(review_text, criteria)
    return reward


def scalar_feedback_metric(example, pred, trace=None, *args, **kwargs):
    """Provide scalar feedback for GEPA optimization."""
    reward = compute_score_metric(example, pred, trace)
    # GEPA expects a score between 0 and 1, which our reward already provides
    return reward


def evaluate_model(model, dataset, model_name):
    """Evaluate a model on the given dataset."""
    print(f"\n=== Evaluating {model_name} ===")

    eval_dataset = dataset

    evaluator = Evaluate(
        devset=eval_dataset,
        metric=compute_score_metric,
        num_threads=72,
        display_progress=True,
    )

    eval_result = evaluator(model)
    score = eval_result.score
    results = [entry[2] for entry in eval_result.results]

    # Calculate standard error
    reward_se = (
        statistics.stdev(results) / (len(results) ** 0.5) if len(results) > 1 else 0
    )

    print(f"Average reading level appropriateness: {score:.4f}±{reward_se:.4f}")

    return score


def main():
    print("Loading reviews reading level data...")
    trainset, valset = load_reviews_reading_level_data()
    print(
        f"Loaded {len(trainset)} training examples, {len(valset)} validation examples"
    )

    model = ReviewModule()

    print("Running GEPA optimization...")

    # Use subset for optimization
    random.shuffle(trainset)
    train_subset = trainset[:400]
    val_subset = trainset[400:450]
    eval_subset = valset

    gepa = GEPA(
        metric=scalar_feedback_metric,
        max_metric_calls=36000,
        # auto="medium",  # Use medium mode for optimization
        num_threads=72,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=dspy.LM(
            model="openai/gpt-4o-mini", temperature=1.0, max_tokens=4000
        ),
        log_dir="gepa_logs_reviews_level",
        use_wandb=True,
        wandb_init_kwargs={
            "entity": "bare-sky",
            "project": "advisor-models-baselines",
            "name": "reviews_level_gepa",
        },
        wandb_api_key=os.getenv("WANDB_API_KEY"),
    )

    optimized_model = gepa.compile(model, trainset=train_subset, valset=val_subset)
    print("Optimized prompt:")
    for name, pred in optimized_model.named_predictors():
        print("================================")
        print(f"Predictor: {name}")
        print("================================")
        print("Prompt:")
        print(pred.signature.instructions)
        print("*********************************")

    print("Evaluating optimized model...")
    evaluate_model(
        optimized_model,
        eval_subset,
        "GEPA-Optimized Reading Level Review Generator",
    )


if __name__ == "__main__":
    main()
