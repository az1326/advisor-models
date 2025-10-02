import os
import re
import json
import argparse
import numpy as np
import openai
# import anthropic
# import vertexai

from tqdm import tqdm

# from google.auth import default, transport
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from gen_payer import generate_taxpayer
from micro_evaluation import (
    rule_list,
    compute_answer,
    analyze_response,
    compute_metrics,
    aggregate_rule_wise_metrics,
)
from structured_forms import TaxPayer
from prompt import (
    basic_forms,
    basic_forms_textual,
    itemized_forms,
    self_employ_forms,
    edu_forms,
    schedule_8812,
)
from prompt_distractor import basic_forms_distractor
from one_shot_example import example_dict

from dotenv import load_dotenv

load_dotenv()

system_prompt = "You are a helpful US taxation consultant."

prompt_template = """
You are given several forms used to report US income tax and the instructions or rules about how to fill the forms. Then you will be given the income and/or payment information about a tax payer According to the given information. You should calculate the income tax owed by this payer.
$example
IRS Forms for the tax payer:
$forms
Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:
"""

tbd_mark = "[__]"


def llama(model, inputs):
    engine = openai.OpenAI(
        base_url=f"https://{MODEL_LOCATION}-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{MODEL_LOCATION}/endpoints/openapi/chat/completions?",
        api_key=credentials.token,
    )
    response = engine.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inputs},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def qwen(model, inputs):
    api_key = os.environ["QWEN_API_KEY"]
    engine = openai.OpenAI(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", api_key=api_key
    )
    response = engine.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inputs},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def gpt(model, inputs):
    api_key = os.environ["OPENAI_API_KEY"]
    engine = openai.OpenAI(api_key=api_key)
    response = engine.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": inputs},
        ],
        temperature=0.0,
    )
    # response.usage.prompt_tokens
    return response.choices[0].message.content


def o1(model, inputs):
    api_key = os.environ["OPENAI_API_KEY"]
    engine = openai.OpenAI(api_key=api_key)
    response = engine.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": system_prompt + "\n\n" + inputs}],
    )
    return response.choices[0].message.content


def claude(model, inputs):
    api_key = os.environ["CLAUDE_API_KEY"]
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        temperature=0.0,
        system=system_prompt,
        max_tokens=8192,
        messages=[{"role": "user", "content": [{"type": "text", "text": inputs}]}],
    )
    return response.content[0].text


def load_problems(complexity: int):
    with open(f"synthesized_problems/comp_{complexity}.json", "r") as f:
        problems = json.load(f)
    return problems


def build_prompt(tax_payer: dict):
    if not args.distractor:
        forms = [basic_forms if not args.textual else basic_forms_textual]
        if tax_payer["itemized"]:
            forms.append(itemized_forms)
        if tax_payer["self_employed"]:
            forms.append(self_employ_forms)
        if tax_payer["has_student_loans_or_education_expenses"]:
            forms.append(edu_forms)
        if tax_payer["child_and_dependent"]:
            forms.append(schedule_8812)
        forms = "".join(forms)
    else:
        forms = basic_forms_distractor
    tbd_fields = []
    for k, v in tax_payer["data"].items():
        forms = forms.replace("$" + k, "$" + f"{v:,}" if not isinstance(v, str) else v)
        if v == "$TBD":
            tbd_fields.append(k)
    for fields in tbd_fields:
        tax_payer["data"].pop(fields)
    forms = forms.replace("$TBD", tbd_mark)
    prompt = prompt_template.replace("$forms", forms)

    prompt = prompt.replace("$name", tax_payer["name"])
    prompt = prompt.replace("$age", str(tax_payer["age"]))
    prompt = prompt.replace("$spouse_age", str(tax_payer["spouse_age"]))
    prompt = prompt.replace("$blind", str(tax_payer["blind"]))
    prompt = prompt.replace("$spouse_blind", str(tax_payer["spouse_blind"]))
    prompt = prompt.replace("$filing_status", tax_payer["filing_status"])
    prompt = prompt.replace("$itemized", str(tax_payer["itemized"]))
    prompt = prompt.replace(
        "$num_qualifying_children", str(tax_payer["num_qualifying_children"])
    )
    prompt = prompt.replace(
        "$num_other_dependents", str(tax_payer["num_other_dependents"])
    )

    return prompt


def eval_query(
    llm, tax_payer: dict, tax_payer_pydantic: TaxPayer, args, query_idx: int
):
    error_cnt = 0
    while True:
        try:
            error_cnt += 1
            prompt, acc = build_prompt(tax_payer), False
            if args.use_example:
                prompt = prompt.replace("$example", example_dict[args.llm])
            else:
                prompt = prompt.replace("$example", "")
            if args.placeholder:
                prompt = prompt.replace("\n\n", "\n" + "- " * 30 + "-\n")

            response = llm(args.llm, prompt)
            response = response.replace("**", "")

            pattern = r"The total tax (owed|overpaid) is \$((?:\d{1,3}(?:,\d{3})*|\d+)(\.\d+)?)."
            match = re.search(pattern, response)
            if match:
                status = match.group(1)
                value = float(match.group(2).replace(",", ""))
                value = -value if status == "overpaid" else value
                answer, _ = compute_answer(tax_payer_pydantic)
                if np.isclose(value, answer) or (
                    isinstance(value, str) and str(answer) in value
                ):
                    acc = True

            rule_applications, structured_response = analyze_response(
                response, tax_payer, tax_payer_pydantic
            )
            break
        except Exception as e:
            print(e)
            if error_cnt >= 5:
                with open(os.path.join(log_path, f"{query_idx}.log"), "w") as f:
                    f.write(response + "\n\n")
                    f.write(f"GPT Predicted: {value}\nRule Calculated: {answer}\n\n")
                return False, []
            continue

    with open(os.path.join(log_path, f"{query_idx}.log"), "w") as f:
        f.write(tax_payer_pydantic.model_dump_json(indent=2) + "\n\n")
        f.write("===\n")
        f.write(prompt + response)
        f.write("\n===\n")
        if match:
            f.write(f"GPT Predicted: {value}; Actual: {answer}\n")
            f.write("===\n")
        f.write("Structured Response:\n")
        f.write(structured_response.model_dump_json(indent=2))
        f.write("\n===\n")
        f.write("Rule Applications:\n")
        f.write("\n".join(rule_applications))

    return acc, rule_applications


parser = argparse.ArgumentParser()
parser.add_argument(
    "--llm",
    type=str,
    choices=[
        "gpt-4o-2024-08-06",
        "claude-3-5-sonnet-20241022",
        "qwen2.5-72b-instruct",
        "meta/llama-3.1-405b-instruct-maas",
        "meta/llama-3.1-70b-instruct-maas",
        "o1-preview",
        "gpt-4o-mini",
        "gpt-4o",
    ],
)
parser.add_argument(
    "--complexity", type=int, default=0, choices=[0, 1, 2]
)  # Difficulty level
parser.add_argument(
    "--use_example", action="store_true"
)  # Whether to use 1-shot example
parser.add_argument(
    "--textual", action="store_true"
)  # Whether to convert tabular rules into textual rules
parser.add_argument(
    "--distractor", action="store_true"
)  # Whether to add distractive rules (NOTE: only use when complexity = 0)
parser.add_argument(
    "--placeholder", action="store_true"
)  # Whether to add meaningless tokens (NOTE: only use when complexity = 0)
parser.add_argument(
    "--chunk_size", type=int, default=1
)  # Number of API Request sent in parallel
parser.add_argument(
    "--log_dir", type=str, default="logs"
)  # Directory to store the experiment logs
parser.add_argument("--remake_data", action="store_true")  # Re-create synthesized data
parser.add_argument(
    "--start_idx", type=int, default=0
)  # Resume experiments from a given question id
parser.add_argument(
    "--eval_size", type=int, default=100
)  # Number of questions to evaluate
args = parser.parse_args()

if args.llm.startswith("o1"):
    llm = o1
elif args.llm.startswith("gpt"):
    llm = gpt
elif args.llm.startswith("claude"):
    llm = claude
elif args.llm.startswith("meta"):
    MODEL_LOCATION = "us-central1"
    PROJECT_ID = "vast-art-443608-e4"
    BUCKET_NAME = "llama-airlines"
    BUCKET_URI = f"gs://{BUCKET_NAME}"

    vertexai.init(
        project=PROJECT_ID, location=MODEL_LOCATION, staging_bucket=BUCKET_URI
    )
    credentials, _ = default()
    auth_request = transport.requests.Request()
    credentials.refresh(auth_request)

    llm = llama
elif args.llm.lower().startswith("qwen"):
    llm = qwen
else:
    raise NotImplementedError(f"LLM ({args.llm}) not implemented")

os.makedirs(args.log_dir, exist_ok=True)
log_path = (
    f"{args.llm}_comp_{args.complexity}_1shot"
    if args.use_example
    else f"{args.llm}_comp_{args.complexity}"
)
if args.textual:
    log_path += "_textual"
if args.distractor:
    log_path += "_distractor"
elif args.placeholder:
    log_path += "_placeholder"
log_path = os.path.join(args.log_dir, log_path)
os.makedirs(log_path, exist_ok=True)

os.makedirs("synthesized_problems", exist_ok=True)
problem_file = os.path.join("synthesized_problems", f"comp_{args.complexity}.json")
if args.remake_data or not os.path.exists(problem_file):
    problem_set = []
    for _ in range(100):
        tax_payer_dict, tax_payer_pydantic = generate_taxpayer(
            complexity=args.complexity
        )
        problem_set.append(
            {"dict": tax_payer_dict, "pydantic": tax_payer_pydantic.model_dump()}
        )
    with open(problem_file, "w") as f:
        json.dump(problem_set, f, indent=2)

problems = load_problems(complexity=args.complexity)[: args.eval_size]

total_acc, total = 0, args.start_idx
rule_applications = []

if args.start_idx >= 1:
    for i in range(args.start_idx):
        tax_payer, tax_payer_pydantic = (
            problems[i]["dict"],
            TaxPayer(**problems[i]["pydantic"]),
        )
        if not os.path.exists(os.path.join(log_path, f"{i}.log")):
            continue
        with open(os.path.join(log_path, f"{i}.log"), "r") as f:
            lines = f.readlines()
        for j in range(len(lines)):
            if lines[j].startswith("Your response:"):
                break
        for k in range(j + 1, len(lines)):
            if lines[k].startswith("==="):
                break
        response = "".join(lines[j + 1 : k])
        pattern = (
            r"The total tax (owed|overpaid) is \$((?:\d{1,3}(?:,\d{3})*|\d+)(\.\d+)?)."
        )
        match = re.search(pattern, response)
        if match:
            status = match.group(1)
            value = float(match.group(2).replace(",", ""))
            value = -value if status == "overpaid" else value
            answer, _ = compute_answer(tax_payer_pydantic)
            if np.isclose(value, answer) or (
                isinstance(value, str) and str(answer) in value
            ):
                total_acc += 1
        for j in range(len(lines)):
            if lines[j].startswith("Rule Applications:"):
                break
        if j != len(lines) and any(
            [
                lines[j + 1].startswith(tag)
                for tag in ["Correct: ", "Missing: ", "Error: "]
            ]
        ):
            rule_app_checklist = "".join(lines[j + 1 :]).split("\n")
            rule_applications.append(rule_app_checklist)

with tqdm(total=len(problems) - args.start_idx) as t:
    for i in range(args.start_idx, len(problems), args.chunk_size):
        chunk = problems[i : min(i + args.chunk_size, len(problems))]
        with ThreadPoolExecutor() as executor:
            iterator = executor.map(
                lambda p: eval_query(
                    llm, p[1]["dict"], TaxPayer(**p[1]["pydantic"]), args, p[0]
                ),
                zip(range(i, min(i + args.chunk_size, len(problems))), chunk),
            )
            for acc, rule_app_checklist in iterator:
                total_acc += acc
                total += 1
                rule_applications.append(rule_app_checklist)
        t.update(len(chunk))

# compute metrics
problem_wise_recall, problem_wise_precision = 0, 0
rule_wise_app_list = defaultdict(lambda: [])
for rule_app_checklist in rule_applications:
    if len(rule_app_checklist) == 0:
        continue
    problem_wise_binary, problem_wise_ratio, rule_wise = compute_metrics(
        rule_app_checklist
    )
    problem_wise_recall += problem_wise_ratio["recall"]
    problem_wise_precision += problem_wise_ratio["precision"]
    for r, app_list in rule_wise.items():
        rule_wise_app_list[r].extend(app_list)
rule_wise_recall, rule_wise_precision, rule_wise_total = aggregate_rule_wise_metrics(
    rule_wise_app_list
)

problem_wise_recall /= len(problems)
problem_wise_precision /= len(problems)

with open(os.path.join(log_path, "overall.log"), "w") as f:
    f.write(f"{total_acc}, {total}\n")
    f.write(f"Accuracy: {(total_acc / total):.4f}\n")
    f.write(f"Problem-Wise Recall: {problem_wise_recall:.4f}\n")
    f.write(f"Problem-Wise Precision: {problem_wise_precision:.4f}\n")
    for r in rule_list:
        f.write(f"Recall of {r}: {rule_wise_recall[r]:.4f}\n")
        f.write(f"Precision of {r}: {rule_wise_precision[r]:.4f}\n")
        f.write(f"Total Trigger of {r}: {rule_wise_total[r]:d}\n")
