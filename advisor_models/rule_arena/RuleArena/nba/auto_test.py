import os
import json
import argparse
import numpy as np
import openai
import anthropic

from tqdm import tqdm, trange
from micro_evaluation import RuleExtraction, parse_rule_application
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

system_prompt = "You are a helpful NBA team consultant."

prompt_template = """
You are given rules in NBA Collective Bargaining Agreement and the information about some teams and players. Then you will be given a list of operations, each of which desribes how some teams conduct some transaction. You should determine whether each operation complies with the given rules.

Assume:
* the Salary Cap for the prior (2023-24) Salary Cap Year is $136,000,000;
* the Average Player Salary for the prior (2023-24) Salary Cap Year is $9,700,000;
* the Salary Cap for the current (2024-25) NBA Salary Cap Year is $140,588,000;
* the Luxury Tax is $170,814,000;
* the First Apron Level is $178,132,000;
* the Second Apron Level is $188,931,000;
* the Team Salary of each team listed under "Team Situations:" do not include the amount of contracts that expire at the end of 2023-2024 Salary Cap Year.

Reference Rules in NBA Collective Bargaining Agreement:

$reference_rules
$example
Decide whether any operation by any team violate the rules:

$question

Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter as A/B/C/...
Your response:
"""

prompt_example = """
Here is an example for you to follow:
<example>
Decide whether operations of any team violate the rules:

Team Situations:
Team A has a team salary of $100,000,000.

Player Situations:
Player A was the 3rd first-round pick of Team A in 2009 NBA draft when he was 20 years old.
Player A signed a 3-year contract (annual salary $22,000,000, 5% increase per year) with Team B during 2021 Moratorium Period.
Player B was the 15th first-round pick of Team B in 2014 NBA draft when he was 19 years old.
Player B signed a 2-year contract (annual salary $10,000,000, 5% increase per year) with Team A during 2022 Moratorium Period.

Operations:
A. Team A signs a 4-year contract with Player A providing annual salary $21,000,000 in the first Salary Cap Year (2024-2025) and 5% increase per year.
B. Team A signs a 2-year contract with Player B providing annual salary $8,000,000 in the first Salary Cap Year (2024-2025) and 5% increase per year.

Analyze the described operations and explicitly state the type of Salary Cap Exceptions if you think the exception should be involved. Conclude your response with:
* "Answer: False." if there is no violation to the rules;
* "Answer: True. Illegal Operation: X. Problematic Team: Y." if Team Y in Operation X violates the rules. Both X and Y should be a single capital letter as A/B/C/...
Your response:
To determine whether any of the operations violate the rules, we must analyze each operation and see if any team involved in them is violating the NBA Collective Bargaining Agreement (CBA) rules.

Team Situations:
- Team A: Has a team salary of $100,000,000 (excluding expiring contracts at the end of the 2023-2024 Salary Cap Year).

Player Situations:
- Player A:
  - Drafted 3rd overall by Team A in 2009 at age 20.
  - Signed a 3-year contract during the 2021 Moratorium Period with starting salary of $22,000,000 and 5% annual increases.
- Player B:
  - Drafted 15th overall by Team B in 2014 at age 20.
  - Signed a 2-year contract during the 2022 Moratorium Period with starting salary of $10,000,000 and 5% annual increases.

Operations:

Operation A: Team A signs Player A to a 4-year contract starting at $21,000,000 with 5% annual increases.
- Team A's Situation:
  - Current Team Salary: $100,000,000.
  - Player B's 2-year contract for Team A just expired, so Player B is an Early-Qualifying Veteran Free Agent of Team A.
  - Player B's prior salary: $10,000,000 × 105% = $10,500,000.
  - Player B has not signed a new contract yet. According to Article VII, Section 4(d)(2), Player B will be included at one hundred thirty percent (130%) of his prior Salary: $10,500,000 × 130% = $13.65 million.
  - Therefore, Team A's actual Team Salary should be $100,000,000 + $13,650,000 = $113,650,000.
- Player A's Situation:
  - Current Age: 20 + (2024 - 2009) = 35 years old.
  - Years of Service: 2024 - 2009 = 15 years.
  - Completed his last contract in Team B, hence Team A has no bird rights of Player A.
- Maximum Salary Calculation (10 or more Years of Service):
  - 35% of Salary Cap ($140,588,000) = $49,205,800.
  - Proposed starting salary of $21,000,000 is within the maximum.
- Salary Cap Space Consumption:
  - The new contract covers 4 seasons. After the new contract ends, the player will be 35 + 4 = 39 years old.
  - Therefore, the salary of the fourth Salary Cap Year (2027-2028) should be attributed to the prior Salary Cap Years pro rata on the basis of the Salaries for such prior Salary Cap Years.
  - Salary in each Salary Cap Year:
    - 2024-2025: $21,000,000
    - 2025-2026: $21,000,000 + $21,000,000 × 5% = $22,050,000
    - 2026-2027: $22,050,000 + $21,000,000 × 5% = $23,100,000
    - 2027-2028: $23,100,000 + $21,000,000 × 5% = $24,150,000
  - Attribute the salary of 2027-2028 Salary Cap Year to the first three Salary Cap Years:
    - 2024-2025: $21,000,000 + $24,150,000 × ($21,000,000 / ($21,000,000 + $22,050,000 + $23,100,000)) ≈ $28.67 million.
    - 2025-2026: $22,050,000 + $24,150,000 × ($22,050,000 / ($21,000,000 + $22,050,000 + $23,100,000)) = $30.10 million.
    - 2026-2027: $23,100,000 + $24,150,000 × ($23,100,000 / ($21,000,000 + $22,050,000 + $23,100,000)) ≈ $31.53 million.
  - After signing Player A, Team A's Team Salary for 2024-2025 Salary Cap Year becomes $113,650,000 + $28.67 million = $142.32 million.
  - Team A's Team Salary after signing exceeds the Salary Cap, it must use a Salary Cap Exception to sign Player A.
  - Team A's Team Salary before signing is below the Salary Cap, so the only exception it can use is the Mid-Level Salary Exceptio (MLE) for Room Teams.
  - The MLE for Room Teams allows a maximum salary of $140,588,000 × 5.678% ≈ $7.98 million < $28.67 million.
  - Therefore, Team A cannot use the MLE for Room Teams to sign Player A, and Operation A violates the rules.

Operation B: Team A signs Player B to a 2-year contract starting at $8,000,000 with 5% annual increases.
- Team A's Situation After Operation A:
  - Team Salary: $142.32 million
- Player B's Situation:
  - Current Age: 20 + (2024 - 2014) = 30 years old.
  - Years of Service: 2024 - 2014 = 10 years.
  - Completed his last 2-year contract in Team A, hence Player B is an Early-Qualifying Veteran Free Agent of Team A.
- Maximum Salary Calculation (10 or more Years of Service):
  - 35% of Salary Cap ($140,588,000) = $49,205,800.
  - Proposed starting salary of $8,000,000 is within the maximum.
- Player B does not trigger the Over 38 Rule, so the Salary for 2024-2025 Salary Cap Year is $8,000,000.
- Team A has exceeded the Salary Cap, so it must use a Salary Cap Exception to sign Player B:
  - Team A has the early Bird rights for Player B, which allows a maximum salary as the greater of:
    - 175% of the Regular Salary for the final Salary Cap Year covered by his prior Contract: 175% × $10,500,000 = $18.375 million.
    - 105% of the Average Player Salary for the prior Salary Cap Year: 105% × $9,700,000 = $10.185 million.
  - The greater is $18.375 million > $8,000,000, and thus Team A can use its early Bird rights to re-sign Player B.
  
Conclusion:
  - Team A cannot sign Player A to a 4-year contract starting at $21,000,000 with 5% annual increases as it would exceed the Salary Cap.
  - Team A can re-sign Player B using its early Bird rights.
  
Answer: True. Illegal Operation: A. Problematic Team: A.
</example>
"""

import vertexai
from google.auth import default, transport


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
        max_tokens=4096,
        temperature=0.0,
        system=system_prompt,
        messages=[{"role": "user", "content": [{"type": "text", "text": inputs}]}],
    )
    return response.content[0].text


def load_problems(complexity: int):
    with open(f"annotated_problems/comp_{complexity}.json", "r") as f:
        problems = json.load(f)
    return problems


def build_query_prompt(query_dict: dict):
    team_info = "Team Situations:\n" + "\n".join(query_dict["team_situations"])
    player_info = "Player Situations:\n" + "\n".join(query_dict["player_situations"])
    operations = "Operations:\n" + "\n".join(query_dict["operations"])
    query_prompt = team_info + "\n\n" + player_info + "\n\n" + operations
    return query_prompt


def eval_query(llm, query_dict: dict, query_idx: int, args):
    query_prompt = build_query_prompt(query_dict)
    prompt = prompt_template.replace("$reference_rules", rules)
    prompt = prompt.replace("$question", query_prompt)
    if args.use_example:
        prompt = prompt.replace("$example", prompt_example)
    else:
        prompt = prompt.replace("$example", "")
    error_cnt = 0
    while True:
        try:
            error_cnt += 1
            response = llm(args.llm, prompt)
            response = response.replace("**", "")

            answer = query_dict["answer"]
            if not answer:
                target_response = "Answer: False"
            else:
                illegal_operation = query_dict["illegal_operation"]
                problematic_team = query_dict["problematic_team"]
                target_response = f"Answer: True. Illegal Operation: {illegal_operation}. Problematic Team: {problematic_team}"

            acc = target_response in response
            applied_rule_list = sorted(parse_rule_application(query_prompt, response))
            ground_truth_rule_list = sorted(query_dict["relevant_rules"])
            break
        except Exception as e:
            print(e)
            if error_cnt >= 5:
                with open(os.path.join(log_path, f"{query_idx}.log"), "w") as f:
                    f.write(response + "\n\n----------LLM Response Ends----------\n\n")
                    f.write(target_response + ".\n\n")
                return False, [], ground_truth_rule_list
            continue

    with open(os.path.join(log_path, f"{query_idx}.log"), "w") as f:
        f.write(response + "\n\n----------LLM Response Ends----------\n\n")
        f.write(target_response + ".\n\n")
        f.write("Parsed Rules:\n" + "\n".join(applied_rule_list) + "\n\n")
        f.write("True Rules:\n" + "\n".join(ground_truth_rule_list))

    return acc, applied_rule_list, ground_truth_rule_list


with open("reference_rules.txt", "r") as f:
    rules = "".join(f.readlines())

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
    ],
)
parser.add_argument(
    "--complexity", type=int, default=0, choices=[0, 1, 2]
)  # Difficulty level
parser.add_argument(
    "--use_example", action="store_true"
)  # Whether to use 1-shot example
parser.add_argument(
    "--chunk_size", type=int, default=1
)  # Number of API Request sent in parallel
parser.add_argument(
    "--log_dir", type=str, default="logs"
)  # Directory to store the experiment logs
parser.add_argument(
    "--start_idx", type=int, default=0
)  # Resume experiments from a given question id
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
    raise NotImplementedError(f"LLM not implemented: {args.llm}")

os.makedirs(args.log_dir, exist_ok=True)
log_path = (
    f"{args.llm}_comp_{args.complexity}_1shot"
    if args.use_example
    else f"{args.llm}_comp_{args.complexity}"
)
log_path = os.path.join(args.log_dir, log_path)
os.makedirs(log_path, exist_ok=True)

problems = load_problems(args.complexity)

total_acc, total = 0, args.start_idx
rule_applications = []

# NOTE: Continue experiment if interrupted
if args.start_idx >= 1:
    for i in trange(args.start_idx):
        if not os.path.exists(os.path.join(log_path, f"{i}.log")):
            print(i)
            continue
        with open(os.path.join(log_path, f"{i}.log"), "r") as f:
            lines = f.readlines()
        for j in range(len(lines)):
            if lines[j].startswith("----------LLM Response Ends----------"):
                break
        llm_response = "".join(lines[: j - 1])
        pred_rules = []
        j += 5
        while not lines[j + 1].startswith("True Rules:"):
            pred_rules.append(lines[j].replace("\n", ""))
            j += 1
        answer = problems[i]["answer"]
        if not answer:
            target_response = "Answer: False"
        else:
            illegal_operation = problems[i]["illegal_operation"]
            problematic_team = problems[i]["problematic_team"]
            target_response = f"Answer: True. Illegal Operation: {illegal_operation}. Problematic Team: {problematic_team}"
        total_acc += target_response in llm_response

        true_rules = sorted(problems[i]["relevant_rules"])
        rule_applications.append((pred_rules, true_rules))

with tqdm(total=len(problems) - args.start_idx) as t:
    for i in range(args.start_idx, len(problems), args.chunk_size):
        chunk = problems[i : min(i + args.chunk_size, len(problems))]
        with ThreadPoolExecutor() as executor:
            iterator = executor.map(
                lambda p: eval_query(llm, p[1], p[0], args),
                zip(range(i, i + len(chunk)), chunk),
            )
            for acc, applied_rule_list, ground_truth_rule_list in iterator:
                total_acc += acc
                total += 1
                rule_applications.append((applied_rule_list, ground_truth_rule_list))
        t.update(len(chunk))

# NOTE: Add metric computation
problem_wise_recall, problem_wise_precision = [], []
tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
for applied_rule_list, ground_truth_rule_list in rule_applications:
    problem_wise_tp, problem_wise_fp, problem_wise_fn = 0, 0, 0
    for rule in set(applied_rule_list + ground_truth_rule_list):
        if rule in applied_rule_list and rule in ground_truth_rule_list:
            tp[rule] += 1
            problem_wise_tp += 1
        elif rule in applied_rule_list:
            fp[rule] += 1
            problem_wise_fp += 1
        else:
            fn[rule] += 1
            problem_wise_fn += 1
    if problem_wise_tp + problem_wise_fn > 0:
        problem_wise_recall.append(
            problem_wise_tp / (problem_wise_tp + problem_wise_fn)
        )
    if problem_wise_tp + problem_wise_fp > 0:
        problem_wise_precision.append(
            problem_wise_tp / (problem_wise_tp + problem_wise_fp)
        )

problem_wise_recall = np.mean(problem_wise_recall)
problem_wise_precision = np.mean(problem_wise_precision)

all_rules = RuleExtraction.model_fields
rule_wise_recall, rule_wise_precision, rule_wise_total = dict(), dict(), dict()
for rule in all_rules:
    if tp[rule] + fp[rule] > 0:
        rule_wise_precision[rule] = tp[rule] / (tp[rule] + fp[rule])
    else:
        rule_wise_precision[rule] = np.nan
    if tp[rule] + fn[rule] > 0:
        rule_wise_recall[rule] = tp[rule] / (tp[rule] + fn[rule])
    else:
        rule_wise_recall[rule] = np.nan
    rule_wise_total[rule] = tp[rule] + fp[rule] + fn[rule]

with open(os.path.join(log_path, "overall.log"), "w") as f:
    f.write(f"{total_acc}, {total}\n")
    f.write(f"Accuracy: {(total_acc / total):.4f}\n")
    f.write(f"Problem-Wise Recall: {problem_wise_recall:.4f}\n")
    f.write(f"Problem-Wise Precision: {problem_wise_precision:.4f}\n")
    for r in all_rules:
        f.write(f"Recall of {r}: {rule_wise_recall[r]:.4f}\n")
        f.write(f"Precision of {r}: {rule_wise_precision[r]:.4f}\n")
        f.write(f"Total Trigger of {r}: {rule_wise_total[r]:d}\n")
