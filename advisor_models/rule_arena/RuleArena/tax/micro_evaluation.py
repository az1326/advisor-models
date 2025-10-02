import os
import numpy as np
import pandas as pd
import openai

from typing import Union
from prompt import (
    basic_forms,
    itemized_forms,
    self_employ_forms,
    edu_forms,
    schedule_8812,
)
from structured_forms import Form1040, TaxPayer, Student

system_prompt = "You are a helpful US taxation consultant an expert at structured data extraction. You are given a tax payer's information and a LLM's response to fill out the 1040 form step-by-step according to the information provided. You should convert the unstructured LLM response into the given structure."

prompt_template = """
IRS Forms for the tax payer:
$forms
Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:
$response
"""

tbd_mark = "[__]"

rule_list = [
    "wage and tip compensation total",
    "net profit",
    "self employment tax",
    "self employment deductible",
    "additional income",
    "total income",
    "total adjustments",
    "adjusted gross income",
    "standard deductions",
    "itemized deductions",
    "total deductions",
    "taxable income",
    "standard taxes",
    "qualified dividends and capital gains taxes",
    "schedule 2 total part i taxes",
    "schedule 2 taxes copy",
    "accumulated taxes",
    "ctc or other dependent credit",
    "additional child tax credit",
    "american opportunity credit",
    "education credits",
    "schedule 3 line 8",
    "schedule 3 total credits copy",
    "accumulated credits computation",
    "taxes after credits computation",
    "other taxes computation",
    "other additional taxes copy",
    "total tax computation",
    "total other payments and refundable credits computation",
    "total payments computation",
    "amount owed or overpaid computation",
]


def build_prompt(tax_payer: dict, response: str):
    forms = [basic_forms]
    if tax_payer["itemized"]:
        forms.append(itemized_forms)
    if tax_payer["self_employed"]:
        forms.append(self_employ_forms)
    if tax_payer["has_student_loans_or_education_expenses"]:
        forms.append(edu_forms)
    if tax_payer["child_and_dependent"]:
        forms.append(schedule_8812)
    forms = "".join(forms)
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
    prompt = prompt.replace("$response", response)

    return prompt


def parse(tax_payer: dict, inputs: str):
    api_key = os.environ["OPENAI_API_KEY"]
    prompt = build_prompt(tax_payer, inputs)
    engine = openai.OpenAI(api_key=api_key)
    response = engine.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        response_format=Form1040,
        temperature=0.0,
    )
    return response.choices[0].message.parsed


def standard_deduction(
    filing_status: str, age: int, spouse_age: int, blind: bool, spouse_blind: bool
):
    if filing_status == "single":
        box_checked = (age >= 65) + blind
        if box_checked == 2:
            return 17550
        elif box_checked == 1:
            return 15700
        else:
            return 13850
    if filing_status == "married filing jointly":
        box_checked = (age >= 65) + blind + (spouse_age >= 65) + spouse_blind
        if box_checked == 4:
            return 33700
        elif box_checked == 3:
            return 32200
        elif box_checked == 2:
            return 30700
        elif box_checked == 1:
            return 29200
        else:
            return 27700
    if filing_status == "head of household":
        box_checked = (age >= 65) + blind
        if box_checked == 2:
            return 24500
        elif box_checked == 1:
            return 22650
        else:
            return 20800
    if filing_status == "married filing separately":
        box_checked = (age >= 65) + blind
        if box_checked == 2:
            return 16850
        elif box_checked == 1:
            return 15350
        else:
            return 13850
    if filing_status == "qualifying surviving spouse":
        box_checked = (age >= 65) + blind
        if box_checked == 2:
            return 30700
        elif box_checked == 1:
            return 29200
        else:
            return 27700


def itemized_deduction(tax_payer: Union[TaxPayer, Form1040]):
    medical_dental_expenses = tax_payer.medical_dental_expenses
    total_adjusted_gross_income = tax_payer.adjusted_gross_income
    sche_a_line_4 = max(
        medical_dental_expenses - 0.075 * total_adjusted_gross_income, 0
    )
    if tax_payer.filing_status == "married filing separately":
        sche_a_line_5e = min(
            tax_payer.state_local_income_or_sales_tax
            + tax_payer.state_local_real_estate_tax
            + tax_payer.state_local_personal_property_tax,
            5000,
        )
    else:
        sche_a_line_5e = min(
            tax_payer.state_local_income_or_sales_tax
            + tax_payer.state_local_real_estate_tax
            + tax_payer.state_local_personal_property_tax,
            10000,
        )
    sche_a_line_7 = tax_payer.other_taxes_paid + sche_a_line_5e
    sche_a_line_8 = (
        tax_payer.home_mortgage_interest_and_points
        + tax_payer.home_mortgage_interest_unreported
        + tax_payer.home_mortgage_points_unreported
    )
    sche_a_line_10 = sche_a_line_8 + tax_payer.investment_interest
    sche_a_line_14 = tax_payer.charity_cash + tax_payer.charity_non_cash
    sche_a_line_17 = (
        sche_a_line_4
        + sche_a_line_7
        + sche_a_line_10
        + sche_a_line_14
        + tax_payer.casualty_and_theft_loss
        + tax_payer.other_itemized_deductions
    )
    return sche_a_line_17


def tax_table(filing_status: str, taxable_income: int):
    if filing_status == "single":
        cuts = [0, 11000, 44725, 95375, 182100, 231250, 578125, 1e20]
    if filing_status == "married filing separately":
        cuts = [0, 11000, 44725, 95375, 182100, 231250, 346875, 1e20]
    if (
        filing_status == "married filing jointly"
        or filing_status == "qualifying surviving spouse"
    ):
        cuts = [0, 22000, 89450, 190750, 364200, 462500, 693750, 1e20]
    if filing_status == "head of household":
        cuts = [0, 15700, 59850, 95350, 182100, 231250, 578100, 1e20]

    rate = [0.1, 0.12, 0.22, 0.24, 0.32, 0.35, 0.37]
    i = 0
    total = 0
    while taxable_income >= cuts[i]:
        total = total + (min(taxable_income, cuts[i + 1]) - cuts[i]) * rate[i]
        i = i + 1
    return total


def calc_tax(taxable_income: int, filing_status: str):
    if taxable_income == 0:
        return 0
    if taxable_income >= 100000:
        return tax_table(filing_status, taxable_income)
    return tax_table(filing_status, np.floor(taxable_income / 50) * 50 + 25)


def qualified_dividends_and_capital_gain_tax_worksheet(
    tax_payer: Union[TaxPayer, Form1040],
):
    qdcg_worksheet_line_1 = tax_payer.computed_taxable_income
    qdcg_worksheet_line_2 = tax_payer.qualified_dividends
    qdcg_worksheet_line_3 = 0
    qdcg_worksheet_line_4 = qdcg_worksheet_line_2 + qdcg_worksheet_line_3
    qdcg_worksheet_line_5 = max(qdcg_worksheet_line_1 - qdcg_worksheet_line_4, 0)
    if (
        tax_payer.filing_status == "single"
        or tax_payer.filing_status == "married filing separately"
    ):
        qdcg_worksheet_line_6 = 44625
    elif (
        tax_payer.filing_status == "married filing jointly"
        or tax_payer.filing_status == "qualifying surviving spouse"
    ):
        qdcg_worksheet_line_6 = 89250
    else:
        qdcg_worksheet_line_6 = 59750
    qdcg_worksheet_line_7 = min(qdcg_worksheet_line_1, qdcg_worksheet_line_6)
    qdcg_worksheet_line_8 = min(qdcg_worksheet_line_5, qdcg_worksheet_line_7)
    qdcg_worksheet_line_9 = qdcg_worksheet_line_7 - qdcg_worksheet_line_8
    qdcg_worksheet_line_10 = min(qdcg_worksheet_line_1, qdcg_worksheet_line_4)
    qdcg_worksheet_line_11 = qdcg_worksheet_line_9
    qdcg_worksheet_line_12 = qdcg_worksheet_line_10 - qdcg_worksheet_line_11
    if tax_payer.filing_status == "single":
        qdcg_worksheet_line_13 = 492300
    elif tax_payer.filing_status == "married filing separately":
        qdcg_worksheet_line_13 = 276900
    elif (
        tax_payer.filing_status == "married filing jointly"
        or tax_payer.filing_status == "qualifying surviving spouse"
    ):
        qdcg_worksheet_line_13 = 553850
    else:
        qdcg_worksheet_line_13 = 523050
    qdcg_worksheet_line_14 = min(qdcg_worksheet_line_1, qdcg_worksheet_line_13)
    qdcg_worksheet_line_15 = qdcg_worksheet_line_5 + qdcg_worksheet_line_9
    qdcg_worksheet_line_16 = max(qdcg_worksheet_line_14 - qdcg_worksheet_line_15, 0)
    qdcg_worksheet_line_17 = min(qdcg_worksheet_line_12, qdcg_worksheet_line_16)
    qdcg_worksheet_line_18 = qdcg_worksheet_line_17 * 0.15
    qdcg_worksheet_line_19 = qdcg_worksheet_line_9 + qdcg_worksheet_line_17
    qdcg_worksheet_line_20 = qdcg_worksheet_line_10 - qdcg_worksheet_line_19
    qdcg_worksheet_line_21 = qdcg_worksheet_line_20 * 0.20
    qdcg_worksheet_line_22 = calc_tax(qdcg_worksheet_line_5, tax_payer.filing_status)
    qdcg_worksheet_line_23 = (
        qdcg_worksheet_line_18 + qdcg_worksheet_line_21 + qdcg_worksheet_line_22
    )
    qdcg_worksheet_line_24 = calc_tax(qdcg_worksheet_line_1, tax_payer.filing_status)
    qdcg_worksheet_line_25 = min(qdcg_worksheet_line_23, qdcg_worksheet_line_24)
    return qdcg_worksheet_line_25


def credit_limit_worksheet(tax_payer: Union[TaxPayer, Form1040]):
    ws_a_line_1 = tax_payer.f1040_line_18
    ws_a_line_2 = (
        tax_payer.foreign_tax_credit
        + tax_payer.dependent_care
        + tax_payer.computed_education_credits
        + tax_payer.retirement_savings
        + tax_payer.elderly_disabled_credits
    )
    ws_a_line_3 = ws_a_line_1 - ws_a_line_2
    ws_a_line_5 = ws_a_line_3
    return ws_a_line_5


def earned_income_worksheet(tax_payer: Union[TaxPayer, Form1040]):
    line_1a = tax_payer.wage_tip_compensation_total
    line_1b = tax_payer.nontaxable_combat_pay
    line_2a = 0  # NOTE: No statutory employee income for Schedule C Line 1
    line_2b = tax_payer.net_profit
    line_3 = line_1a + line_1b + line_2a + line_2b
    if line_3 <= 0:
        return 0
    line_4 = 0
    line_5 = tax_payer.self_employment_deductible
    line_6 = line_4 + line_5
    line_7 = line_3 - line_6
    return line_7


def compute_child_and_dependent_credits(tax_payer: Union[TaxPayer, Form1040]):
    line_1 = tax_payer.adjusted_gross_income
    line_4 = tax_payer.num_qualifying_children
    line_5 = line_4 * 2000
    line_6 = tax_payer.num_other_dependents
    line_7 = line_6 * 500
    line_8 = line_5 + line_7
    if tax_payer.filing_status == "married filing jointly":
        line_9 = 400000
    else:
        line_9 = 200000
    line_10 = max(line_1 - line_9, 0)
    line_10 = np.ceil(line_10 / 1000) * 1000
    line_11 = line_10 * 0.05
    line_12 = line_8 - line_11
    if line_12 <= 0:
        return 0, 0

    line_13 = credit_limit_worksheet(tax_payer)
    line_14 = min(line_12, line_13)
    line_16a = line_12 - line_14
    if line_16a <= 0:
        return line_14, 0

    line_16b = line_4 * 1600
    if line_16b == 0:
        return line_14, 0

    line_17 = min(line_16a, line_16b)
    line_18a = earned_income_worksheet(tax_payer)
    # line_18b = tax_payer.nontaxable_combat_pay
    line_19 = max(line_18a - 2500, 0)
    line_20 = line_19 * 0.15
    if line_4 < 3:
        line_27 = min(line_17, line_20)
    elif line_20 >= line_17:
        line_27 = line_17
    else:
        line_22 = tax_payer.self_employment_deductible
        line_24 = 0  # EIC and Schedule 3, line 11 are 0 in current version
        line_25 = max(line_22 - line_24, 0)
        line_26 = max(line_20, line_25)
        line_27 = min(line_17, line_26)

    return line_14, line_27


def compute_education_credits(tax_payer: TaxPayer):
    line_1 = line_10 = 0
    for student in tax_payer.student_list:
        if isinstance(student, Student):
            student = student.model_dump()
        go_to_line_31 = False
        if student["f8863_part_iii_23"] == "Yes":
            go_to_line_31 = True
        elif student["f8863_part_iii_24"] == "No":
            go_to_line_31 = True
        elif student["f8863_part_iii_25"] == "Yes":
            go_to_line_31 = True
        elif student["f8863_part_iii_26"] == "Yes":
            go_to_line_31 = True

        if go_to_line_31:
            line_10 += student["qualified_student_expenses"]
        else:
            line_27 = min(student["qualified_student_expenses"], 4000)
            line_28 = max(line_27 - 2000, 0)
            line_29 = line_28 * 0.25
            if line_28 == 0:
                line_1 += line_27
            else:
                line_1 += line_29 + 2000

    if tax_payer.filing_status == "married filing jointly":
        line_2 = 180000
    else:
        line_2 = 90000
    line_3 = tax_payer.adjusted_gross_income
    line_4 = line_2 - line_3
    if line_4 <= 0:
        return 0, 0

    if tax_payer.filing_status == "married filing jointly":
        line_5 = 20000
    else:
        line_5 = 10000
    if line_4 >= line_5:
        line_6 = 1.000
    else:
        line_6 = round(line_4 / line_5, 3)
    line_7 = line_1 * line_6
    line_8 = line_7 * 0.40 if tax_payer.age >= 24 else 0
    line_9 = line_7 - line_8
    line_11 = min(line_10, 10000)
    line_12 = line_11 * 0.2
    if tax_payer.filing_status == "married filing jointly":
        line_13 = 180000
    else:
        line_13 = 90000
    line_14 = tax_payer.adjusted_gross_income
    line_15 = line_13 - line_14
    if line_15 <= 0:
        line_18 = 0
    else:
        if tax_payer.filing_status == "married filing jointly":
            line_16 = 20000
        else:
            line_16 = 10000
        if line_15 >= line_16:
            line_17 = 1.000
        else:
            line_17 = round(line_15 / line_16, 3)
        line_18 = line_12 * line_17
    clws_line_3 = line_18 + line_9
    clws_line_4 = tax_payer.f1040_line_18
    clws_line_5 = (
        tax_payer.foreign_tax_credit
        + tax_payer.dependent_care
        + tax_payer.elderly_disabled_credits
    )
    clws_line_6 = clws_line_4 - clws_line_5
    line_19 = min(clws_line_3, clws_line_6)
    return line_8, line_19


def compute_answer(tax_payer: TaxPayer):
    tax_payer.wage_tip_compensation_total = (
        tax_payer.wage_tip_compensation
        + tax_payer.household_employee_wage
        + tax_payer.unreported_tip
    )

    if tax_payer.self_employed:
        tax_payer.gross_profit = (
            tax_payer.gross_receipts
            - tax_payer.returns_and_allowances
            - tax_payer.cost_of_goods_sold
        )
        tax_payer.gross_income = tax_payer.gross_profit + tax_payer.other_inc_sched_c
        tax_payer.tentative_profit = tax_payer.gross_income - tax_payer.total_expenses
        tax_payer.net_profit = tax_payer.tentative_profit - tax_payer.expenses_of_home
        tax_payer.sche_se_line_4c = (
            tax_payer.net_profit * 0.9235
            if tax_payer.net_profit > 0
            else tax_payer.net_profit
        )
        if tax_payer.sche_se_line_4c < 400:
            tax_payer.self_employment_tax = tax_payer.self_employment_deductible = 0
        else:
            if tax_payer.total_social_security_wages >= 160200:
                tax_payer.sche_se_line_10 = 0
            else:
                tax_payer.sche_se_line_9 = (
                    160200 - tax_payer.total_social_security_wages
                )
                tax_payer.sche_se_line_10 = (
                    min(tax_payer.sche_se_line_4c, tax_payer.sche_se_line_9) * 0.124
                )
            tax_payer.self_employment_tax = (
                tax_payer.sche_se_line_4c * 0.029 + tax_payer.sche_se_line_10
            )
            tax_payer.self_employment_deductible = tax_payer.self_employment_tax * 0.5
    else:
        tax_payer.gross_receipts = tax_payer.net_profit = (
            tax_payer.self_employment_tax
        ) = tax_payer.self_employment_deductible = 0

    tax_payer.additional_income = (
        tax_payer.taxable_state_refunds
        + tax_payer.alimony_income
        + tax_payer.net_profit
        + tax_payer.sale_of_business
        + tax_payer.rental_real_estate_sch1
        + tax_payer.farm_income
        + tax_payer.unemployment_compensation
        + tax_payer.other_income
    )

    tax_payer.computed_total_income = (
        tax_payer.wage_tip_compensation_total
        + tax_payer.taxable_interest
        + tax_payer.ordinary_dividends
        + tax_payer.taxable_ira_distributions
        + tax_payer.taxable_pensions
        + tax_payer.taxable_social_security_benefits
        + 0  # Capital gains fixed at 0
        + tax_payer.additional_income
    )

    tax_payer.total_adjustments = (
        tax_payer.educator_expenses
        + tax_payer.hsa_deduction
        + tax_payer.self_employment_deductible
        + tax_payer.ira_deduction
        + tax_payer.student_loan_interest_deduction
        + tax_payer.other_adjustments
    )

    tax_payer.adjusted_gross_income = (
        tax_payer.computed_total_income - tax_payer.total_adjustments
    )

    if tax_payer.itemized:
        tax_payer.itemized_deductions = itemized_deduction(tax_payer)
        tax_payer.standard_or_itemized_deductions = tax_payer.itemized_deductions
    else:
        tax_payer.standard_deduction = standard_deduction(
            tax_payer.filing_status,
            tax_payer.age,
            tax_payer.spouse_age,
            tax_payer.blind,
            tax_payer.spouse_blind,
        )
        tax_payer.standard_or_itemized_deductions = tax_payer.standard_deduction

    tax_payer.total_deductions = (
        tax_payer.standard_or_itemized_deductions + tax_payer.qualified_business_income
    )

    tax_payer.computed_taxable_income = max(
        tax_payer.adjusted_gross_income - tax_payer.total_deductions, 0
    )

    if tax_payer.qualified_dividends > 0:  # or tax_payer.capital_gains > 0:
        tax_payer.computed_taxes = qualified_dividends_and_capital_gain_tax_worksheet(
            tax_payer
        )
    else:
        tax_payer.computed_taxes = calc_tax(
            tax_payer.computed_taxable_income, tax_payer.filing_status
        )

    tax_payer.schedule_2_total_taxes = tax_payer.amt_f6251 + tax_payer.credit_repayment

    tax_payer.f1040_line_18 = (
        tax_payer.computed_taxes + tax_payer.schedule_2_total_taxes
    )

    if tax_payer.has_student_loans_or_education_expenses:
        (
            tax_payer.computed_american_opportunity_credit,
            tax_payer.computed_education_credits,
        ) = compute_education_credits(tax_payer)
    else:
        tax_payer.computed_american_opportunity_credit = (
            tax_payer.computed_education_credits
        ) = 0

    tax_payer.schedule_3_line_8 = (
        tax_payer.foreign_tax_credit
        + tax_payer.dependent_care
        + tax_payer.computed_education_credits
        + tax_payer.retirement_savings
        + tax_payer.elderly_disabled_credits
        + tax_payer.plug_in_motor_vehicle
        + tax_payer.alt_motor_vehicle
    )

    if tax_payer.num_qualifying_children + tax_payer.num_other_dependents > 0:
        (
            tax_payer.computed_ctc_or_other_dependent_credit,
            tax_payer.computed_additional_child_tax_credit,
        ) = compute_child_and_dependent_credits(tax_payer)
    else:
        tax_payer.computed_ctc_or_other_dependent_credit = (
            tax_payer.computed_additional_child_tax_credit
        ) = 0

    tax_payer.computed_accumulated_credits = (
        tax_payer.schedule_3_line_8 + tax_payer.computed_ctc_or_other_dependent_credit
    )

    tax_payer.computed_taxes_after_credits = (
        tax_payer.f1040_line_18 - tax_payer.computed_accumulated_credits
    )

    tax_payer.computed_other_taxes = (
        tax_payer.other_additional_taxes + tax_payer.self_employment_tax
    )

    tax_payer.computed_total_tax = (
        tax_payer.computed_taxes_after_credits + tax_payer.computed_other_taxes
    )

    tax_payer.total_other_payments_and_refundable_credits = (
        tax_payer.earned_income_credit
        + tax_payer.computed_additional_child_tax_credit
        + tax_payer.computed_american_opportunity_credit
        + 0
    )  # 0 is the additional child tax credit, the reserved for future use, and the amount from schedule 3 line 15

    tax_payer.total_payments = (
        tax_payer.federal_income_tax_withheld
        + 0
        + tax_payer.total_other_payments_and_refundable_credits
    )  # 0 is the estimated tax payments

    tax_payer.computed_amount_owed_or_overpaid = (
        tax_payer.computed_total_tax - tax_payer.total_payments
    )

    return tax_payer.computed_amount_owed_or_overpaid, tax_payer


def analyze_response(response: str, tax_payer: dict, tax_payer_pydantic: TaxPayer):
    rule_app_stat = []
    structured_output = parse(tax_payer, response)

    name, age, spouse_age, filing_status = (
        structured_output.name,
        structured_output.age,
        structured_output.spouse_age,
        structured_output.filing_status,
    )
    if (
        name != tax_payer_pydantic.name
        or age != tax_payer_pydantic.age
        or spouse_age != tax_payer_pydantic.spouse_age
        or filing_status != tax_payer_pydantic.filing_status
    ):
        rule_app_stat.append("Error: basic info")
    else:
        rule_app_stat.append("Correct: basic info")

    # Check Line 1z - 工资收入合计
    computed_total = (
        structured_output.wage_tip_compensation
        + structured_output.household_employee_wage
        + structured_output.unreported_tip
    )
    if np.isclose(structured_output.wage_tip_compensation_total, computed_total):
        rule_app_stat.append("Correct: wage and tip compensation total")
    elif np.isclose(structured_output.wage_tip_compensation_total, 0):
        rule_app_stat.append("Missing: wage and tip compensation total")
    else:
        rule_app_stat.append("Error: wage and tip compensation total")

    # Check Schedule C & Schedule SE (Schedule 1 Line 3, 15 & Schedule 2 Line 4)
    if tax_payer_pydantic.self_employed:
        if structured_output.gross_receipts is None:
            structured_output.gross_receipts = 0
        if structured_output.net_profit is None:
            structured_output.net_profit = 0
        if structured_output.self_employment_tax is None:
            structured_output.self_employment_tax = 0
        if structured_output.self_employment_deductible is None:
            structured_output.self_employment_deductible = 0
        gross_profit = (
            structured_output.gross_receipts
            - structured_output.returns_and_allowances
            - structured_output.cost_of_goods_sold
        )
        gross_income = gross_profit + structured_output.other_inc_sched_c
        tentative_profit = gross_income - structured_output.total_expenses
        net_profit = tentative_profit - structured_output.expenses_of_home
        sche_se_line_4c = (
            structured_output.net_profit * 0.9235
            if structured_output.net_profit > 0
            else structured_output.net_profit
        )
        if sche_se_line_4c < 400:
            self_employment_tax = self_employment_deductible = 0
        else:
            if structured_output.total_social_security_wages >= 160200:
                sche_se_line_10 = 0
            else:
                sche_se_line_9 = 160200 - structured_output.total_social_security_wages
                sche_se_line_10 = min(sche_se_line_4c, sche_se_line_9) * 0.124
            self_employment_tax = sche_se_line_4c * 0.029 + sche_se_line_10
            self_employment_deductible = structured_output.self_employment_tax * 0.5
        if structured_output.net_profit is not None and np.isclose(
            structured_output.net_profit, net_profit
        ):
            rule_app_stat.append("Correct: net profit")
        elif structured_output.net_profit is None or np.isclose(
            structured_output.net_profit, 0
        ):
            rule_app_stat.append("Missing: net profit")
        else:
            rule_app_stat.append("Error: net profit")
        if structured_output.self_employment_tax is not None and np.isclose(
            structured_output.self_employment_tax, self_employment_tax
        ):
            rule_app_stat.append("Correct: self employment tax")
        elif (
            np.isclose(structured_output.self_employment_tax, 0)
            or structured_output.self_employment_tax is None
        ):
            rule_app_stat.append("Missing: self employment tax")
        else:
            rule_app_stat.append("Error: self employment tax")
        if not rule_app_stat[-1] == "Missing: self employment tax":
            if structured_output.self_employment_deductible is not None and np.isclose(
                structured_output.self_employment_deductible, self_employment_deductible
            ):
                rule_app_stat.append("Correct: self employment deductible")
            elif (
                np.isclose(structured_output.self_employment_deductible, 0)
                or structured_output.self_employment_deductible is None
            ):
                rule_app_stat.append("Missing: self employment deductible")
            else:
                rule_app_stat.append("Error: self employment deductible")
    else:
        if structured_output.gross_receipts is None:
            structured_output.gross_receipts = 0
        if structured_output.net_profit is None:
            structured_output.net_profit = 0
        if structured_output.self_employment_tax is None:
            structured_output.self_employment_tax = 0
        if structured_output.self_employment_deductible is None:
            structured_output.self_employment_deductible = 0

    # Check Line 8 (Schedule 1 Line 10) - additional income
    computed_additional_income = (
        structured_output.taxable_state_refunds
        + structured_output.alimony_income
        + structured_output.net_profit
        + structured_output.sale_of_business
        + structured_output.rental_real_estate_sch1
        + structured_output.farm_income
        + structured_output.unemployment_compensation
        + structured_output.other_income
    )
    if np.isclose(structured_output.additional_income, computed_additional_income):
        rule_app_stat.append("Correct: additional income")
    elif np.isclose(structured_output.additional_income, 0):
        rule_app_stat.append("Missing: additional income")
    else:
        rule_app_stat.append("Error: additional income")

    # Check Line 9 - total income
    computed_total_income = (
        structured_output.wage_tip_compensation_total
        + structured_output.taxable_interest
        + structured_output.ordinary_dividends
        + structured_output.taxable_ira_distributions
        + structured_output.taxable_pensions
        + structured_output.taxable_social_security_benefits
        + 0  # Capital gains fixed at 0
        + structured_output.additional_income
    )
    if np.isclose(structured_output.total_income, computed_total_income):
        rule_app_stat.append("Correct: total income")
    elif np.isclose(structured_output.total_income, 0):
        rule_app_stat.append("Missing: total income")
    else:
        rule_app_stat.append("Error: total income")

    # Check Line 10 (Schedule 1 Line 26) - total adjustment
    computed_total_adjustments = (
        structured_output.educator_expenses
        + structured_output.hsa_deduction
        + structured_output.self_employment_deductible
        + structured_output.ira_deduction
        + structured_output.student_loan_interest_deduction
        + structured_output.other_adjustments
    )
    if np.isclose(structured_output.total_adjustments, computed_total_adjustments):
        rule_app_stat.append("Correct: total adjustments")
    elif np.isclose(structured_output.total_adjustments, 0):
        rule_app_stat.append("Missing: total adjustments")
    else:
        rule_app_stat.append("Error: total adjustments")

    # Check Line 11 - adjusted gross income (AGI)
    computed_agi = structured_output.total_income - structured_output.total_adjustments
    if np.isclose(structured_output.adjusted_gross_income, computed_agi):
        rule_app_stat.append("Correct: adjusted gross income")
    elif np.isclose(structured_output.adjusted_gross_income, 0):
        rule_app_stat.append("Missing: adjusted gross income")
    else:
        rule_app_stat.append("Error: adjusted gross income")

    # Check Line 12 - standard / itemized deduction
    if structured_output.itemized:
        computed_standard_or_itemized_deductions = itemized_deduction(structured_output)
        if np.isclose(
            structured_output.standard_or_itemized_deductions,
            standard_deduction(
                structured_output.filing_status,
                structured_output.age,
                structured_output.spouse_age,
                structured_output.blind,
                structured_output.spouse_blind,
            ),
        ):
            rule_app_stat.append("Missing: itemized deductions")
        elif np.isclose(
            structured_output.standard_or_itemized_deductions,
            computed_standard_or_itemized_deductions,
        ):
            rule_app_stat.append("Correct: itemized deductions")
        else:
            rule_app_stat.append("Error: itemized deductions")
    else:
        computed_standard_or_itemized_deductions = standard_deduction(
            structured_output.filing_status,
            structured_output.age,
            structured_output.spouse_age,
            structured_output.blind,
            structured_output.spouse_blind,
        )
        if np.isclose(
            structured_output.standard_or_itemized_deductions,
            computed_standard_or_itemized_deductions,
        ):
            rule_app_stat.append("Correct: standard deductions")
        elif np.isclose(structured_output.standard_or_itemized_deductions, 0):
            rule_app_stat.append("Missing: standard deductions")
        else:
            rule_app_stat.append("Error: standard deductions")

    # Check Line 14 - total deductions
    computed_total_deductions = (
        structured_output.standard_or_itemized_deductions
        + structured_output.qualified_business_income
    )
    if np.isclose(structured_output.total_deductions, computed_total_deductions):
        rule_app_stat.append("Correct: total deductions")
    elif np.isclose(structured_output.total_deductions, 0):
        rule_app_stat.append("Missing: total deductions")
    else:
        rule_app_stat.append("Error: total deductions")

    # Check Line 15 - taxable income
    computed_taxable_income = max(
        structured_output.adjusted_gross_income - structured_output.total_deductions, 0
    )
    if np.isclose(structured_output.computed_taxable_income, computed_taxable_income):
        rule_app_stat.append("Correct: taxable income")
    elif np.isclose(structured_output.computed_taxable_income, 0):
        rule_app_stat.append("Missing: taxable income")
    else:
        rule_app_stat.append("Error: taxable income")

    # Check Line 16 - tax (with / without qualified dividends)
    if (
        structured_output.qualified_dividends > 0
    ):  # or structured_output.capital_gains > 0:
        computed_taxes = qualified_dividends_and_capital_gain_tax_worksheet(
            structured_output
        )
        if np.isclose(structured_output.taxes, computed_taxes):
            rule_app_stat.append("Correct: qualified dividends and capital gains taxes")
        elif np.isclose(
            structured_output.taxes,
            calc_tax(
                structured_output.computed_taxable_income,
                structured_output.filing_status,
            ),
        ) or np.isclose(structured_output.taxes, 0):
            rule_app_stat.append("Missing: qualified dividends and capital gains taxes")
        else:
            rule_app_stat.append("Error: qualified dividends and capital gains taxes")
    else:
        computed_taxes = calc_tax(
            structured_output.computed_taxable_income, structured_output.filing_status
        )
        if np.isclose(structured_output.taxes, computed_taxes):
            rule_app_stat.append("Correct: standard taxes")
        elif np.isclose(structured_output.taxes, 0):
            rule_app_stat.append("Missing: standard taxes")
        else:
            rule_app_stat.append("Error: standard taxes")

    # Check Schedule 2 Line 3
    if structured_output.schedule_2_total_taxes is not None and np.isclose(
        structured_output.schedule_2_total_taxes,
        structured_output.amt_f6251 + structured_output.credit_repayment,
    ):
        rule_app_stat.append("Correct: schedule 2 total part i taxes")
    elif structured_output.schedule_2_total_taxes is None or np.isclose(
        structured_output.schedule_2_total_taxes, 0
    ):
        rule_app_stat.append("Missing: schedule 2 total part i taxes")
    else:
        rule_app_stat.append("Error: schedule 2 total part i taxes")

    # Check Line 17 - Copy Schedule 2 Line 3
    copy_alternative_minimum_tax = structured_output.schedule_2_total_taxes
    if np.isclose(
        structured_output.copy_schedule_2_line_3, copy_alternative_minimum_tax
    ):
        rule_app_stat.append("Correct: schedule 2 taxes copy")
    elif np.isclose(structured_output.copy_schedule_2_line_3, 0):
        rule_app_stat.append("Missing: schedule 2 taxes copy")
    else:
        rule_app_stat.append("Error: schedule 2 taxes copy")

    # Check Line 18
    computed_f1040_line_18 = (
        structured_output.taxes + structured_output.copy_schedule_2_line_3
    )
    if np.isclose(structured_output.f1040_line_18, computed_f1040_line_18):
        rule_app_stat.append("Correct: accumulated taxes")
    elif np.isclose(structured_output.f1040_line_18, 0):
        rule_app_stat.append("Missing: accumulated taxes")
    else:
        rule_app_stat.append("Error: accumulated taxes")

    # Check Line 19 & 28 - CTC tax credit and additional child tax credit
    if (
        structured_output.num_qualifying_children
        + structured_output.num_other_dependents
        > 0
    ):
        computed_ctc_or_other_dependent_credit, computed_additional_child_tax_credit = (
            compute_child_and_dependent_credits(structured_output)
        )
        if np.isclose(
            structured_output.ctc_or_other_dependent_credit,
            computed_ctc_or_other_dependent_credit,
        ):
            rule_app_stat.append("Correct: ctc or other dependent credit")
        elif np.isclose(structured_output.ctc_or_other_dependent_credit, 0):
            rule_app_stat.append("Missing: ctc or other dependent credit")
        else:
            rule_app_stat.append("Error: ctc or other dependent credit")
        if np.isclose(
            structured_output.additional_child_tax_credit,
            computed_additional_child_tax_credit,
        ):
            rule_app_stat.append("Correct: additional child tax credit")
        elif np.isclose(structured_output.additional_child_tax_credit, 0):
            rule_app_stat.append("Missing: additional child tax credit")
        else:
            rule_app_stat.append("Error: additional child tax credit")

    # Check Line 29 & Schedule 3 Line 3 - education tax credits
    if (
        structured_output.student_list is not None
        and len(structured_output.student_list) > 0
    ):
        american_opportunity_credit, education_credits = compute_education_credits(
            structured_output
        )
        if np.isclose(
            structured_output.american_opportunity_credit, american_opportunity_credit
        ):
            rule_app_stat.append("Correct: american opportunity credit")
        elif np.isclose(structured_output.american_opportunity_credit, 0):
            rule_app_stat.append("Missing: american opportunity credit")
        else:
            rule_app_stat.append("Error: american opportunity credit")
        if np.isclose(structured_output.computed_education_credits, education_credits):
            rule_app_stat.append("Correct: education credits")
        elif np.isclose(structured_output.computed_education_credits, 0):
            rule_app_stat.append("Missing: education credits")
        else:
            rule_app_stat.append("Error: education credits")
    else:
        american_opportunity_credit = 0

    # Check Schedule 3 Line 8
    target_schedule_3_line_8 = (
        structured_output.foreign_tax_credit
        + structured_output.dependent_care
        + structured_output.computed_education_credits
        + structured_output.retirement_savings
        + structured_output.elderly_disabled_credits
        + structured_output.plug_in_motor_vehicle
        + structured_output.alt_motor_vehicle
    )
    if np.isclose(structured_output.schedule_3_line_8, target_schedule_3_line_8):
        rule_app_stat.append("Correct: schedule 3 line 8")
    elif np.isclose(structured_output.schedule_3_line_8, 0):
        rule_app_stat.append("Missing: schedule 3 line 8")
    else:
        rule_app_stat.append("Error: schedule 3 line 8")

    # Check Line 20 - Schedule 3 Line 8
    copy_total_credits = structured_output.schedule_3_line_8
    if np.isclose(structured_output.copy_schedule_3_line_8, copy_total_credits):
        rule_app_stat.append("Correct: schedule 3 total credits copy")
    elif np.isclose(structured_output.copy_schedule_3_line_8, 0):
        rule_app_stat.append("Missing: schedule 3 total credits copy")
    else:
        rule_app_stat.append("Error: schedule 3 total credits copy")

    # Check Line 21 - total credits
    computed_accumulated_credits = (
        structured_output.copy_schedule_3_line_8
        + structured_output.ctc_or_other_dependent_credit
    )
    if np.isclose(structured_output.accumulated_credits, computed_accumulated_credits):
        rule_app_stat.append("Correct: accumulated credits computation")
    elif np.isclose(structured_output.accumulated_credits, 0):
        rule_app_stat.append("Missing: accumulated credits computation")
    else:
        rule_app_stat.append("Error: accumulated credits computation")

    # Check Line 22
    computed_taxes_after_credits = max(
        structured_output.f1040_line_18 - structured_output.accumulated_credits, 0
    )
    if np.isclose(structured_output.taxes_after_credits, computed_taxes_after_credits):
        rule_app_stat.append("Correct: taxes after credits computation")
    elif np.isclose(structured_output.taxes_after_credits, 0):
        rule_app_stat.append("Missing: taxes after credits computation")
    else:
        rule_app_stat.append("Error: taxes after credits computation")

    # Check Line 23 - other taxes
    if structured_output.self_employment_tax is None:
        structured_output.self_employment_tax = 0
    computed_other_taxes = (
        structured_output.other_additional_taxes + structured_output.self_employment_tax
    )
    if np.isclose(structured_output.schedule_2_total_other_taxes, computed_other_taxes):
        rule_app_stat.append("Correct: other taxes computation")
    elif np.isclose(structured_output.schedule_2_total_other_taxes, 0):
        rule_app_stat.append("Missing: other taxes computation")
    else:
        rule_app_stat.append("Error: other taxes computation")
    copy_other_additional_taxes = structured_output.schedule_2_total_other_taxes
    if np.isclose(structured_output.other_taxes, copy_other_additional_taxes):
        rule_app_stat.append("Correct: other additional taxes copy")
    elif np.isclose(structured_output.other_taxes, 0):
        rule_app_stat.append("Missing: other additional taxes copy")
    else:
        rule_app_stat.append("Error: other additional taxes copy")

    # Check Line 24 - total taxes
    computed_total_tax = (
        structured_output.taxes_after_credits + structured_output.other_taxes
    )
    if np.isclose(structured_output.total_tax, computed_total_tax):
        rule_app_stat.append("Correct: total tax computation")
    elif np.isclose(structured_output.total_tax, 0):
        rule_app_stat.append("Missing: total tax computation")
    else:
        rule_app_stat.append("Error: total tax computation")

    # Check Line 32 - other payments
    computed_total_other_payments = (
        structured_output.earned_income_credit
        + structured_output.additional_child_tax_credit
        + structured_output.american_opportunity_credit
        + structured_output.copy_schedule_3_line_15
    )
    if np.isclose(
        structured_output.total_other_payments_and_refundable_credits,
        computed_total_other_payments,
    ):
        rule_app_stat.append(
            "Correct: total other payments and refundable credits computation"
        )
    elif np.isclose(structured_output.total_other_payments_and_refundable_credits, 0):
        rule_app_stat.append(
            "Missing: total other payments and refundable credits computation"
        )
    else:
        rule_app_stat.append(
            "Error: total other payments and refundable credits computation"
        )

    # Check Line 33 - total payments
    computed_total_payments = (
        structured_output.federal_income_tax_withheld
        + 0
        + structured_output.total_other_payments_and_refundable_credits
    )
    if np.isclose(structured_output.total_payments, computed_total_payments):
        rule_app_stat.append("Correct: total payments computation")
    elif np.isclose(structured_output.total_payments, 0):
        rule_app_stat.append("Missing: total payments computation")
    else:
        rule_app_stat.append("Error: total payments computation")

    # Check Line 37 - taxes owed or overpaid (answer)
    computed_amount_owed_or_overpaid = (
        structured_output.total_tax - structured_output.total_payments
    )
    if np.isclose(
        structured_output.amount_owed_or_overpaid, computed_amount_owed_or_overpaid
    ):
        rule_app_stat.append("Correct: amount owed or overpaid computation")
    elif np.isclose(structured_output.amount_owed_or_overpaid, 0):
        rule_app_stat.append("Missing: amount owed or overpaid computation")
    else:
        rule_app_stat.append("Error: amount owed or overpaid computation")

    return rule_app_stat, structured_output


def compute_metrics(rule_app_checklist: list[str]):
    rule_wise = {r: [] for r in rule_list}
    problem_wise_binary = {"precision": 1, "recall": 1}
    correct, missing, error = 0, 0, 0
    for tag in rule_app_checklist:
        tag_type = tag[: tag.find(": ")]
        tag_rule = tag[tag.find(": ") + 2 :]
        if tag_rule in rule_wise:
            rule_wise[tag_rule].append(tag_type)
            if tag_type == "Missing":
                problem_wise_binary["recall"] = 0
                missing += 1
            elif tag_type == "Error":
                problem_wise_binary["precision"] = 0
                error += 1
            else:
                correct += 1
    problem_wise_ratio = {
        "precision": correct / (correct + error),
        "recall": correct / (correct + missing),
    }
    return problem_wise_binary, problem_wise_ratio, rule_wise


def aggregate_rule_wise_metrics(rule_wise_app_list: dict[str, list[str]]):
    rule_wise_recall, rule_wise_precision, rule_wise_total = (
        {r: np.nan for r in rule_list},
        {r: np.nan for r in rule_list},
        {r: 0 for r in rule_list},
    )
    for rule, app_list in rule_wise_app_list.items():
        if len(app_list) == 0:
            continue
        value_cnt = pd.value_counts(app_list)
        rule_wise_recall[rule] = (len(app_list) - value_cnt.get("Missing", 0)) / len(
            app_list
        )
        if value_cnt.get("Correct", 0) + value_cnt.get("Error", 0) > 0:
            rule_wise_precision[rule] = value_cnt.get("Correct", 0) / (
                value_cnt.get("Correct", 0) + value_cnt.get("Error", 0)
            )
        else:
            rule_wise_precision[rule] = np.nan  # NOTE: fail to recall
        rule_wise_total[rule] = len(app_list)
    return rule_wise_recall, rule_wise_precision, rule_wise_total
