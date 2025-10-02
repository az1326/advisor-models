import os
import json
import argparse
import openai
# import anthropic
# import vertexai

from tqdm import tqdm

# from google.auth import default, transport
from collections import defaultdict
from structure import rule_list
from gen_questions import gen_question
from concurrent.futures import ThreadPoolExecutor
from compute_answer import load_checking_fee, compute_answer
from micro_evaluation import (
    check_rule_application,
    compute_metrics,
    aggregate_rule_wise_metrics,
)

from dotenv import load_dotenv

load_dotenv()

system_prompt = "You are a helpful assistant at American Airlines. You are given the information of a passenger, his / her items, his / her special needs, and the policies of American Airlines. You should compute the total cost (including the flight ticket fee, checked bag fees, cost of special needs) according to the policies for the passenger."

prompt_template = """
The policies of American Airlines are as follows:

$reference_rules
$example_prompt
$question_prompt Compute the total cost for him step by step (don't omit any bag) and end your response with "The total cost is $xxx." (xxx is a number)
Your response:
"""

example = """
Here is an example for you to follow:
<example>
John is a Main Plus Class passenger flying from Orlando to Lima with the following items:
1. A backpack: 21 x 11 x 7 inches, 9 lbs;
2. A luggage box: 51 x 24 x 27 inches, 65 lbs;
3. A backpack: 46 x 29 x 24 inches, 85 lbs;
4. A luggage box: 35 x 12 x 11 inches, 44 lbs;
5. A luggage box: 30 x 11 x 10 inches, 33 lbs;
John's flight ticket is $448. Compute the total cost for him step by step and end your response with "The total cost is $xxx." (xxx is a number)
Your response:
To calculate the total cost for John, we need to consider the flight ticket cost, checked bag fees, and any overweight or oversize fees according to American Airlines' policies.
                                                    
Flight Ticket Cost:                           
- John's flight ticket: $448
                                                    
Carry-on Allowance:                 
- John is allowed 1 carry-on bag and 1 personal item.
- Item 1 (Backpack):                               
  - Dimensions: 21 x 11 x 7 inches
  - Weight: 9 lbs                                 
  - This fits within the carry-on size limit of 22 x 14 x 9 inches.
  - Fee: $0 (Allowed as carry-on bag)
- John does not have any items that fit the personal item dimensions (18 x 14 x 8 inches). So, no personal item is carried.
                                                    
Checked Bags:                  
- Items to be checked: Items 2, 3, 4, and 5
- John is a Main Plus passenger, which includes 1 extra free checked bag in addition to the Main Cabin allowance, for a total of 2 free checked bags.
- Checked Bag Fees:                                                                                     
  - First Bag: $0 (free)
  - Second Bag: $0 (free)                                                                               
  - Third Bag: $200
  - Fourth Bag: $200
                                                    
Fees for Each Checked Bag:
                                                    
1. Item 2 (Luggage box):           
   - Dimensions: 51 x 24 x 27 inches
     - Total dimensions: 51 + 24 + 27 = 102 inches
     - Over the standard size limit of 62 inches.
   - Weight: 65 lbs
     - Over the standard weight limit of 50 lbs but under 70 lbs.
   - Checking Fee:
     - For the first checked bag, the checking fee is $0.
   - Oversize Fee:                                 
     - For dimensions over 65 inches up to 115 inches between the U.S. and South America, the fee is $150.
   - Overweight Fee:
     - For weights over 53 lbs up to 70 lbs, the fee is $100.
   - The higher of oversize and overweight fee should apply.
   - Total Fee for Item 2: $0 (checking) + $150 (oversize) = $150
2. Item 3 (Backpack):
   - Dimensions: 46 x 29 x 24 inches
     - Total dimensions: 46 + 29 + 24 = 99 inches
     - Over the standard size limit of 62 inches.
   - Weight: 85 lbs
     - Over the standard weight limit of 50 lbs and over 70 lbs but under 100 lbs.
   - Checking Fee:
     - For the second checked bag, the checking fee is $0.
   - Oversize Fee:
     - For dimensions over 65 inches up to 115 inches between the U.S. and South America, the fee is $150.
   - Overweight Fee:
     - For weights over 70 lbs up to 100 lbs, the fee is $200.
   - The higher of oversize and overweight fee should apply.
   - Total Fee for Item 3: $0 (checking) + $200 (overweight) = $200
3. Item 4 (Luggage box):
   - Dimensions: 35 x 12 x 11 inches
     - Total dimensions: 35 + 12 + 11 = 58 inches
     - Within the standard size limit of 62 inches.
   - Weight: 44 lbs
     - Within the standard weight limit of 50 lbs.
   - Checking Fee:
     - For the third checked bag, the checking fee is $200.
   - Total Fee for Item 4: $200 (checking) + $0 (No overweight or oversize fees) = $200
4. Item 5 (Luggage box):
   - Dimensions: 30 x 11 x 10 inches
     - Total dimensions: 30 + 11 + 10 = 51 inches
     - Within the standard size limit of 62 inches.
   - Weight: 33 lbs
     - Within the standard weight limit of 50 lbs.
   - Checking Fee:
     - For the fourth checked bag, the checking fee is $200.
   - Total Fee for Item 5: $200 (checking) + $0 (No overweight or oversize fees) = $200
Summary of Baggage Fees:
  - Item 2: $200
  - Item 3: $150
  - Item 4: $200
  - Item 5: $200
Total Baggage Fees: $200 (Item 2) + $150 (Item 3) + $200 (Item 4) + $200 (Item 5) = $750
Total Cost:
- Flight Ticket: $448
- Total Baggage Fees: $750
- The total cost is $1,198.
</example>
"""


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
    with open(f"synthesized_problems/comp_{complexity}.jsonl", "r") as f:
        problems = [json.loads(l) for l in f]
    return problems


def eval_query(llm, question_prompt: str, info_dict: dict, query_idx: int, args):
    error_cnt = 0
    rule_file = (
        "reference_rules.txt" if not args.textual else "reference_rules_textual.txt"
    )
    with open(rule_file, "r") as f:
        reference_rules = f.read()
    prompt = prompt_template.replace("$reference_rules", reference_rules)
    prompt = prompt.replace("$question_prompt", question_prompt)
    if args.use_example:
        prompt = prompt.replace("$example_prompt", example)
    else:
        prompt = prompt.replace("$example_prompt", "")
    while True:
        try:
            error_cnt += 1
            acc, pred = False, None
            check_base_tables = load_checking_fee()
            fee, _ = compute_answer(**info_dict, check_base_tables=check_base_tables)

            response = llm(args.llm, prompt)
            response = response.replace("**", "")

            start_idx = response.find("The total cost is")
            if start_idx != -1:
                conclusion = response[start_idx:]
                value_idx = conclusion.find("$")
                if value_idx != -1:
                    value = (
                        conclusion[value_idx + 1 :].replace(",", "").replace(".", "")
                    )
                    if value.isnumeric():
                        value = int(value)
                    pred = value
                    if value == fee or (isinstance(value, str) and str(fee) in value):
                        acc = True

            rule_app_checklist, structured_response = check_rule_application(
                question_prompt, info_dict, response
            )
            break
        except Exception as e:
            print(e)
            if error_cnt >= 5:
                with open(os.path.join(log_path, f"{query_idx}.log"), "w") as f:
                    f.write(response + "\n\n")
                    f.write(f"GPT Predicted: {pred}\nRule Calculated: {fee}\n\n")
                return False, []
            continue

    with open(os.path.join(log_path, f"{query_idx}.log"), "w") as f:
        f.write(response + "\n\n")
        f.write(f"GPT Predicted: {pred}\nRule Calculated: {fee}\n\n")
        f.write(structured_response.model_dump_json(indent=2) + "\n\n")
        f.write("\n".join(rule_app_checklist))

    return acc, rule_app_checklist


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
        "o1",
        "gpt-4o-mini",
    ],
)
parser.add_argument(
    "--complexity", type=int, choices=[0, 1, 2], default=0
)  # Difficulty level
parser.add_argument(
    "--textual", action="store_true"
)  # Whether to convert tabular rules into textual rules
parser.add_argument(
    "--chunk_size", type=int, default=1
)  # Number of API Request sent in parallel
parser.add_argument(
    "--use_example", action="store_true"
)  # Whether to use 1-shot example
parser.add_argument(
    "--log_dir", type=str, default="logs"
)  # Directory to store the experiment logs
parser.add_argument("--remake_data", action="store_true")  # Re-create synthesized data
parser.add_argument(
    "--start_idx", type=int, default=0
)  # Resume experiments from a given question id
parser.add_argument("--eval_size", type=int, default=100)
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
elif args.llm.startswith("qwen"):
    llm = qwen
else:
    raise NotImplementedError(f"{args.llm} not implemented")

shot = "0shot" if not args.use_example else "1shot"
log_path = os.path.join(args.log_dir, f"{args.llm}_comp_{args.complexity}_{shot}")
if args.textual:
    log_path += "_textual"
os.makedirs(log_path, exist_ok=True)

problem_file = os.path.join("synthesized_problems", f"comp_{args.complexity}.jsonl")
if args.remake_data or not os.path.exists(problem_file):
    problem_set = []
    for _ in range(100):
        question_prompt, info_dict = gen_question(complexity=args.complexity)
        problem_set.append({"prompt": question_prompt, "info": info_dict})
    with open(problem_file, "w") as f:
        for x in problem_set:
            json.dump(x, f)
            f.write("\n")

problems = load_problems(complexity=args.complexity)[: args.eval_size]

total_acc, total = 0, args.start_idx
rule_applications = []

# NOTE: Continue experiment if interrupted
if args.start_idx >= 1:
    for i in range(args.start_idx):
        with open(os.path.join(log_path, f"{i}.log"), "r") as f:
            lines = f.readlines()
        for j in range(len(lines)):
            if "GPT Predicted: " in lines[j]:
                break
        pred = lines[j][15:-1]
        for j in range(len(lines)):
            if "Rule Calculated: " in lines[j]:
                break
        truth = lines[j][17:-1]
        if truth in pred:
            total_acc += 1
        for j in range(len(lines)):
            if any(
                [
                    lines[j].startswith(tag)
                    for tag in ["Correct: ", "Missing: ", "Error: "]
                ]
            ):
                break
        if any(
            [lines[j].startswith(tag) for tag in ["Correct: ", "Missing: ", "Error: "]]
        ):
            rule_app_checklist = "".join(lines[j:]).split("\n")
            rule_applications.append(rule_app_checklist)
    print(f"Initialized with {args.start_idx} cases, {total_acc} correct.")

with tqdm(total=len(problems) - args.start_idx) as t:
    for i in range(args.start_idx, len(problems), args.chunk_size):
        chunk = problems[i : min(i + args.chunk_size, len(problems))]
        with ThreadPoolExecutor() as executor:
            iterator = executor.map(
                lambda p: eval_query(llm, p[1]["prompt"], p[1]["info"], p[0], args),
                zip(range(i, min(i + args.chunk_size, len(problems))), chunk),
            )
            for acc, rule_app_checklist in iterator:
                total_acc += acc
                total += 1
                rule_applications.append(rule_app_checklist)
        t.update(len(chunk))

problem_wise_recall, problem_wise_precision = 0, 0
rule_wise_app_list = defaultdict(lambda: [])
for i, rule_app_checklist in enumerate(rule_applications):
    if len(rule_app_checklist) == 0:
        continue
    problem_wise_metrics, rule_wise = compute_metrics(rule_app_checklist)
    problem_wise_recall += problem_wise_metrics["recall"]
    problem_wise_precision += problem_wise_metrics["precision"]
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
