import numpy as np
import random
from prompt import f8863_part_iii
from structured_forms import TaxPayer


def generate_sche_1(status_dict: dict):
    taxable_state_tax_refunds = np.random.randint(0, 2000)
    alimony_income = np.random.randint(0, 5000)
    sale_of_business = np.random.randint(0, 50000)
    farm_income = 0  # TODO: add farm income
    business_income = "$TBD" if status_dict["self_employed"] else 0
    real_estate = 0  # TODO: add real estate income
    unemployment_compensation = np.random.randint(0, 10000)
    other_income = np.random.randint(0, 2000)
    educator_expenses = np.random.randint(0, 250)
    health_saving_account_deduction = np.random.randint(0, 3600)
    self_employment_deductible = "$TBD" if status_dict["self_employed"] else 0
    ira_deduction = np.random.randint(0, 6000)
    student_loan_interest_deduction = np.random.randint(0, 2500)
    other_adjustment = np.random.randint(0, 1000)

    return {
        "taxable_state_refunds": taxable_state_tax_refunds,
        "alimony_income": alimony_income,
        "business_income": business_income,
        "sale_of_business": sale_of_business,
        "rental_real_estate_sch1": real_estate,
        "farm_income": farm_income,
        "unemployment_compensation": unemployment_compensation,
        "other_income": other_income,
        "educator_expenses": educator_expenses,
        "hsa_deduction": health_saving_account_deduction,
        "self_employment_deductible": self_employment_deductible,
        "ira_deduction": ira_deduction,
        "student_loan_interest_deduction": student_loan_interest_deduction,
        "other_adjustments": other_adjustment,
    }


def generate_sche_2():
    alternative_minimum_tax = np.random.randint(0, 5000)
    credit_repayment = np.random.randint(0, 1000)
    other_taxes = np.random.randint(0, 2000)

    return {
        "amt_f6251": alternative_minimum_tax,
        "credit_repayment": credit_repayment,
        "other_additional_taxes": other_taxes,
    }


def generate_sche_3(status_dict: dict):
    foreign_tax_credit = np.random.randint(0, 1000)
    dependent_care_expenses = np.random.randint(0, 3000)
    education_credits = (
        "$TBD" if status_dict["has_student_loans_or_education_expenses"] else 0
    )
    retirement_saving_credit = np.random.randint(0, 2000)
    elderly_or_disabled_credit = np.random.randint(0, 750)
    qualified_electric_vehicle_credit = np.random.randint(0, 7500)
    alternative_fuel_vehicle_refuel_credit = np.random.randint(0, 2000)
    other_credits_or_payments = np.random.randint(0, 1000)

    return {
        "foreign_tax_credit": foreign_tax_credit,
        "dependent_care": dependent_care_expenses,
        "education_credits": education_credits,
        "retirement_savings": retirement_saving_credit,
        "elderly_disabled_credits": elderly_or_disabled_credit,
        "plug_in_motor_vehicle": qualified_electric_vehicle_credit,
        "alt_motor_vehicle": alternative_fuel_vehicle_refuel_credit,
        "other_credits_or_payments": other_credits_or_payments,
    }


def generate_sche_a():
    medical_dental_expenses = np.random.randint(0, 10000)
    real_estate_taxes = np.random.randint(0, 8000)
    state_and_local_taxes = np.random.randint(0, 10000)
    property_taxes = np.random.randint(0, 5000)
    other_taxes = np.random.randint(0, 3000)
    home_mortgage_interest_and_points = np.random.randint(0, 20000)
    home_mortgage_interest_unreported = np.random.randint(0, 5000)
    home_mortgage_points_unreported = np.random.randint(0, 1000)
    investment_interest = np.random.randint(0, 3000)
    charity_cash = np.random.randint(0, 249)
    charity_non_cash = np.random.randint(0, 249)
    casualty_and_theft_loss = np.random.randint(0, 5000)
    other_itemized_deductions = np.random.randint(0, 2000)

    return {
        "medical_dental_expenses": medical_dental_expenses,
        "state_local_income_or_sales_tax": state_and_local_taxes,
        "state_local_real_estate_tax": real_estate_taxes,
        "state_local_personal_property_tax": property_taxes,
        "other_taxes_paid": other_taxes,
        "home_mortgage_interest_and_points": home_mortgage_interest_and_points,
        "home_mortgage_interest_unreported": home_mortgage_interest_unreported,
        "home_mortgage_points_unreported": home_mortgage_points_unreported,
        "investment_interest": investment_interest,
        "charity_cash": charity_cash,
        "charity_non_cash": charity_non_cash,
        "casualty_and_theft_loss": casualty_and_theft_loss,
        "other_itemized_deductions": other_itemized_deductions,
    }


def generate_sche_c(status_dict: dict):
    gross_receipts = np.random.randint(20000, 120000)
    returns_and_allowances = np.random.randint(0, 5000)
    cost_of_goods_sold = np.random.randint(0, 20000)
    other_income = np.random.randint(0, 5000)
    total_expenses = np.random.randint(0, 40000)
    expenses_of_home = np.random.randint(0, 20000)
    # Schedule SE
    total_social_security_wages = np.random.randint(20000, 120000)

    return {
        "gross_receipts": gross_receipts,
        "returns_and_allowances": returns_and_allowances,
        "cost_of_goods_sold": cost_of_goods_sold,
        "other_inc_sched_c": other_income,
        "total_expenses": total_expenses,
        "expenses_of_home": expenses_of_home,
        "total_social_security_wages": total_social_security_wages,
    }


def generate_sche_e():
    rental_advertising = np.random.randint(0, 1000)
    rental_auto_and_travel = np.random.randint(0, 2000)
    rental_clean_and_maintain = np.random.randint(0, 3000)
    rental_fees = np.random.randint(0, 1000)
    rental_mortgage_interest = np.random.randint(0, 20000)
    rental_other_interest = np.random.randint(0, 5000)
    rental_repair_and_supplies = np.random.randint(0, 5000)
    rental_taxes = np.random.randint(0, 8000)
    rental_utilities = np.random.randint(0, 5000)
    rental_other_expenses = np.random.randint(0, 2000)
    rents_received = np.random.randint(0, 50000)
    royalties_received = np.random.randint(0, 5000)
    royalty_expenses = np.random.randint(0, 2000)
    rental_property_value = np.random.randint(50000, 500000)
    ws1_8582_prior_loss = np.random.randint(0, 10000)

    return {
        "rental_advertising": rental_advertising,
        "rental_auto_and_travel": rental_auto_and_travel,
        "rental_clean_and_maintain": rental_clean_and_maintain,
        "rental_fees": rental_fees,
        "rental_mortgage_interest": rental_mortgage_interest,
        "rental_other_interest": rental_other_interest,
        "rental_repair_and_supplies": rental_repair_and_supplies,
        "rental_taxes": rental_taxes,
        "rental_utilities": rental_utilities,
        "rental_other_expenses": rental_other_expenses,
        "rents_received": rents_received,
        "royalties_received": royalties_received,
        "royalty_expenses": royalty_expenses,
        "rental_property_value": rental_property_value,
        "ws1_8582_prior_loss": ws1_8582_prior_loss,
    }


def generate_sche_f():
    gross_income = np.random.randint(0, 200000)
    total_expenses = np.random.randint(0, 150000)
    net_farm_profit = gross_income - total_expenses

    return {
        "gross_income": gross_income,
        "total_expenses": total_expenses,
        "net_farm_profit": max(net_farm_profit, 0),
    }


def generate_f8863():
    num_students = np.random.randint(1, 4)
    f8863_part_iii_list, student_list = [], []
    for _ in range(num_students):
        qualified_student_expenses = np.random.randint(0, 8000)
        use_line_31 = random.choice([True, False])
        if not use_line_31:
            f8863_part_iii_23 = f8863_part_iii_25 = f8863_part_iii_26 = "No"
            f8863_part_iii_24 = "Yes"
        else:
            f8863_part_iii_23 = random.choice(["Yes", "No"])
            f8863_part_iii_24 = random.choice(["Yes", "No"])
            f8863_part_iii_25 = random.choice(["Yes", "No"])
            f8863_part_iii_26 = random.choice(["Yes", "No"])
        student_list.append(
            {
                "qualified_student_expenses": qualified_student_expenses,
                "f8863_part_iii_23": f8863_part_iii_23,
                "f8863_part_iii_24": f8863_part_iii_24,
                "f8863_part_iii_25": f8863_part_iii_25,
                "f8863_part_iii_26": f8863_part_iii_26,
            }
        )

        prompt = f8863_part_iii
        prompt = prompt.replace(
            "$qualified_student_expenses", "$" + f"{qualified_student_expenses:,}"
        )
        prompt = prompt.replace("$f8863_part_iii_23", f8863_part_iii_23)
        prompt = prompt.replace("$f8863_part_iii_24", f8863_part_iii_24)
        prompt = prompt.replace("$f8863_part_iii_25", f8863_part_iii_25)
        prompt = prompt.replace("$f8863_part_iii_26", f8863_part_iii_26)
        f8863_part_iii_list.append(prompt)

    return {
        "num_students": num_students,
        "student_list": student_list,
        "f8863_part_iii": "".join(f8863_part_iii_list),
    }


def generate_basic_f1040(status_dict: dict):
    tax_exempt_interest = np.random.randint(0, 5000)
    taxable_interest = np.random.randint(0, 10000)
    qualified_dividends = (
        np.random.randint(1000, 5000) if status_dict["has_qualified_dividends"] else 0
    )
    ordinary_dividends = np.random.randint(0, 5000)
    ira_distributions = np.random.randint(0, 20000)
    taxable_ira_distributions = round(ira_distributions * random.uniform(0, 1))
    all_pensions = np.random.randint(0, 50000)
    taxable_pensions = round(all_pensions * random.uniform(0, 1))
    social_security_benefits = np.random.randint(0, 30000)
    taxable_social_security_benefits = round(
        social_security_benefits * random.uniform(0, 0.85)
    )
    qualified_business_income = np.random.randint(0, 10000)
    wage_tip_compensation = np.random.randint(24000, 140000)
    household_employee_wage = np.random.randint(0, 2000)
    unreported_tip = np.random.randint(0, 5000)
    american_opportunity_credit = (
        "$TBD" if status_dict["has_student_loans_or_education_expenses"] else 0
    )
    federal_income_tax_withheld = np.random.randint(0, 20000)
    earned_income_credit = 0  # TODO: No earned income credit in current version
    nontaxable_combat_pay = np.random.randint(0, 10000)

    return {
        "tax_exempt_interest": tax_exempt_interest,
        "taxable_interest": taxable_interest,
        "qualified_dividends": qualified_dividends,
        "ordinary_dividends": ordinary_dividends,
        "ira_distributions": ira_distributions,
        "taxable_ira_distributions": taxable_ira_distributions,
        "all_pensions": all_pensions,
        "taxable_pensions": taxable_pensions,
        "social_security_benefits": social_security_benefits,
        "taxable_social_security_benefits": taxable_social_security_benefits,
        "qualified_business_income": qualified_business_income,
        "wage_tip_compensation": wage_tip_compensation,
        "household_employee_wage": household_employee_wage,
        "unreported_tip": unreported_tip,
        "federal_income_tax_withheld": federal_income_tax_withheld,
        "earned_income_credit": earned_income_credit,
        "american_opportunity_credit": american_opportunity_credit,
        "nontaxable_combat_pay": nontaxable_combat_pay,
    }


def generate_taxpayer(complexity: int):
    if complexity == 0:
        qualified_dividends = False
        num_true = 0
    elif complexity == 1:
        qualified_dividends = random.choice([True, False])
        num_true = random.choice([1, 2])
    elif complexity == 2:
        qualified_dividends = True
        num_true = random.choice([3, 4])
    attached_forms = random.sample(range(4), k=num_true)
    itemized = 0 in attached_forms
    self_employ = 1 in attached_forms
    student = 2 in attached_forms
    child_and_dependent = 3 in attached_forms

    if not student:
        filing_status = random.choice(
            [
                "single",
                "married filing separately",
                "qualifying surviving spouse",
                "married filing jointly",
                "head of household",
            ]
        )
    else:
        filing_status = "married filing jointly"

    age = np.random.randint(25, 80)
    spouse_age = age + np.random.randint(-4, 5)
    blind = random.choice([True] + [False] * 19)
    spouse_blind = random.choice([True] + [False] * 19)

    if child_and_dependent:
        num_total_dependent = np.random.randint(1, 5)
        num_qualifying_children = np.random.randint(0, num_total_dependent + 1)
        num_other_dependents = num_total_dependent - num_qualifying_children
        additional_child_tax_credit = ctc_or_other_dependent_credit = "[__]"
    else:
        num_qualifying_children = num_other_dependents = 0
        additional_child_tax_credit = ctc_or_other_dependent_credit = 0

    status_dict = {
        "name": "John",
        "itemized": itemized,
        "self_employed": self_employ,
        "has_qualified_dividends": qualified_dividends,
        "has_student_loans_or_education_expenses": student,
        "child_and_dependent": child_and_dependent,
        "filing_status": filing_status,
        "age": age,
        "spouse_age": spouse_age,
        "blind": blind,
        "spouse_blind": spouse_blind,
        "num_qualifying_children": num_qualifying_children,
        "num_other_dependents": num_other_dependents,
    }

    f1040 = generate_basic_f1040(status_dict)
    sche1_additional_income_and_adjustments = generate_sche_1(status_dict)
    sche2_additional_taxes = generate_sche_2()
    sche3_credits_and_payments = generate_sche_3(status_dict)

    data_dict = {
        "additional_child_tax_credit": additional_child_tax_credit,
        "ctc_or_other_dependent_credit": ctc_or_other_dependent_credit,
    }
    data_dict.update(f1040)
    data_dict.update(sche1_additional_income_and_adjustments)
    data_dict.update(sche2_additional_taxes)
    data_dict.update(sche3_credits_and_payments)

    if itemized:
        sche_a_itemized_deductions = generate_sche_a()
        # TODO: Generate Form 6251, skipped for now
        data_dict.update(sche_a_itemized_deductions)

    if self_employ:
        sche_c_business_profit_or_loss = generate_sche_c(status_dict)
        data_dict.update(sche_c_business_profit_or_loss)

    if student:
        f8863 = generate_f8863()
        data_dict.update(f8863)

    taxpayer = TaxPayer(**data_dict, **status_dict)

    status_dict["data"] = data_dict
    if student:
        status_dict["data"].pop("student_list")

    return status_dict, taxpayer
