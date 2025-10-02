"""
GEPA baseline for math solutions domain

python baselines/math_solutions/math_solutions_baseline.py
"""

import dspy
from dspy.evaluate import Evaluate
from dspy import GEPA
import pandas as pd
import random
import os
import statistics
from openai import OpenAI
from advisor_models.math_solutions.config import (
    STYLE_JUDGE_SYSTEM_PROMPT,
    STYLE_JUDGE_PROMPT,
)


random.seed(42)
llm = dspy.LM("openai/gpt-4o-mini", cache=False, temperature=1.0)
dspy.settings.configure(lm=llm)

prompt_prefix = """You are a math tutor. Present a step-by-step solution to the given math problem that matches the student's preferred learning style. For example, you might consider whether to include multiple solution methods, to ask questions of the student during the solution, to provide very detailed or big-picture explanations, and/or to use visual guides."""


def load_math_solutions_data():
    """Load math solutions data from parquet files."""
    train_path = "data/math_solutions/train.parquet"
    val_path = "data/math_solutions/validation.parquet"

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Math solutions data files not found at {train_path} or {val_path}"
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Convert to DSPy format
    trainset = []
    for _, row in train_df.iterrows():
        student = row["student"]
        problem = row["original_question"]
        judge_criteria = row["reward_spec"]["judge_criteria"]

        example = dspy.Example(
            prompt=prompt_prefix
            + f" You are writing a solution for {student}. Present a solution to the following problem:\n"
            + problem,
            judge_criteria=judge_criteria,
        ).with_inputs("prompt", "judge_criteria")
        trainset.append(example)

    valset = []
    for _, row in val_df.iterrows():
        student = row["student"]
        problem = row["original_question"]
        judge_criteria = row["reward_spec"]["judge_criteria"]

        example = dspy.Example(
            prompt=prompt_prefix
            + f" You are writing a solution for {student}. Present a solution to the following problem:\n"
            + problem,
            judge_criteria=judge_criteria,
        ).with_inputs("prompt", "judge_criteria")
        valset.append(example)

    return trainset, valset


class MathSolver(dspy.Signature):
    """Solve a math problem step by step."""

    prompt = dspy.InputField(desc="The math problem to solve")
    solution = dspy.OutputField(
        desc="The step-by-step solution with final answer in \\boxed{} format"
    )


class MathSolverModule(dspy.Module):
    """Math solver module."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(MathSolver)

    def forward(self, prompt, judge_criteria=None):
        return self.generate(prompt=prompt)


def compute_style_reward(solution: str, judge_criteria) -> float:
    """Compute style matching reward using LLM-as-a-judge."""
    try:
        judge_prompt = STYLE_JUDGE_PROMPT.format(
            judge_criteria=judge_criteria,
            solution=solution,
        )

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": STYLE_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0.0,
        )

        judge_response = response.choices[0].message.content.strip()

        # Parse the three-way response: ACCEPT=1.0, PARTIAL=0.4, REJECT=0.0
        if "ACCEPT" in judge_response:
            return 1.0
        elif "PARTIAL" in judge_response:
            return 0.4
        else:  # REJECT or any other response
            return 0.0

    except Exception as e:
        print(f"Error computing style reward: {e}")
        return 0.0


def compute_score_metric(example, pred, trace=None):
    """Compute the reward score for a prediction."""
    solution_text = pred.solution
    judge_criteria = example.judge_criteria
    reward = compute_style_reward(solution_text, judge_criteria)
    return reward


def scalar_feedback_metric(example, pred, trace=None, *args, **kwargs):
    """Provide scalar feedback for GEPA optimization."""
    reward = compute_score_metric(example, pred, trace)
    # GEPA expects a score between 0 and 1, which our reward already provides
    return reward


def evaluate_model(model, dataset, model_name):
    """Evaluate a model on the given dataset."""
    print(f"\n=== Evaluating {model_name} ===")

    # Evaluate on subset for faster testing
    eval_dataset = random.sample(dataset, min(100, len(dataset)))

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

    print(f"Average style alignment reward: {score:.4f}±{reward_se:.4f}")

    return score


def main():
    print("Loading math solutions data...")
    trainset, valset = load_math_solutions_data()
    print(
        f"Loaded {len(trainset)} training examples, {len(valset)} validation examples"
    )

    model = MathSolverModule()

    print("Running GEPA optimization...")

    # Use subset for optimization
    random.shuffle(trainset)
    train_subset = trainset[:300]
    val_subset = trainset[300:400]
    eval_subset = valset

    gepa = GEPA(
        metric=scalar_feedback_metric,
        max_metric_calls=64000,
        # auto="medium",  # Use medium mode for optimization
        num_threads=72,
        track_stats=True,
        reflection_minibatch_size=3,
        reflection_lm=dspy.LM(
            model="openai/gpt-4o-mini", temperature=1.0, max_tokens=4000
        ),
        log_dir="gepa_logs_math_solutions",
        use_wandb=True,
        wandb_init_kwargs={
            "entity": "bare-sky",
            "project": "advisor-models-baselines",
            "name": "math_solutions_gepa",
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
    evaluate_model(optimized_model, eval_subset, "GEPA-Optimized Math Solver")


if __name__ == "__main__":
    main()
