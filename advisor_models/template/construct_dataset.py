"""Left blank intentionally. This file should implement construction of a
training and validation dataset saved in parquet format. For this example, each
row of the dataset should be in the following format:
{
    "prompt": List[Dict[str, str]],
    "env_class": str,
    "reward_spec": {"ground_truth": float},
    "original_question": str,
}
`prompt` should be the prompt given to the advisor model, in OpenAI messages format.
`env_class` should be "template", the name of the environment class as assigned in main_template.py.
`ground_truth` should be the ground truth value used to calculate score.
`original_question` should be the original question used to format the student prompt.
Other fields may be included and accessed in the environment via the extras field in `__init__`.
"""
