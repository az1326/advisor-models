from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional


class FilingStatus(str, Enum):
    SINGLE = "single"
    MARRIED_FILING_JOINTLY = "married filing jointly"
    MARRIED_FILING_SEPARATELY = "married filing separately"
    HEAD_OF_HOUSEHOLD = "head of household"
    QUALIFYING_SURVIVING_SPOUSE = "qualifying surviving spouse"


class Student(BaseModel):
    qualified_student_expenses: int = Field(description="Form 8863 Part III Line 21")
    f8863_part_iii_23: bool = Field(description="Form 8863 Part III Line 23")
    f8863_part_iii_24: bool = Field(description="Form 8863 Part III Line 24")
    f8863_part_iii_25: bool = Field(description="Form 8863 Part III Line 25")
    f8863_part_iii_26: bool = Field(description="Form 8863 Part III Line 26")


class Form1040(BaseModel):
    name: str = Field(description="Name of taxpayer")
    age: int = Field(description="Age of taxpayer")
    spouse_age: int = Field(description="Age of taxpayer's spouse")
    filing_status: FilingStatus = Field(description="Filing status of taxpayer")
    blind: bool = Field(description="Taxpayer is blind")
    spouse_blind: bool = Field(description="Taxpayer's spouse is blind")
    itemized: bool = Field(description="Taxpayer uses itemized deductions")
    num_qualifying_children: int = Field(description="Number of qualifying children")
    num_other_dependents: int = Field(description="Number of other dependents")
    wage_tip_compensation: float = Field(description="Form 1040 Line 1a")
    household_employee_wage: float = Field(description="Form 1040 Line 1b")
    unreported_tip: float = Field(description="Form 1040 Line 1c")
    nontaxable_combat_pay: float = Field(description="Form 1040 Line 1d")
    wage_tip_compensation_total: float = Field(description="Form 1040 Line 1z")
    tax_exempt_interest: float = Field(description="Form 1040 Line 2a")
    taxable_interest: float = Field(description="Form 1040 Line 2b")
    qualified_dividends: float = Field(description="Form 1040 Line 3a")
    ordinary_dividends: float = Field(description="Form 1040 Line 3b")
    ira_distributions: float = Field(description="Form 1040 Line 4a")
    taxable_ira_distributions: float = Field(description="Form 1040 Line 4b")
    all_pensions: float = Field(description="Form 1040 Line 5a")
    taxable_pensions: float = Field(description="Form 1040 Line 5b")
    social_security_benefits: float = Field(description="Form 1040 Line 6a")
    taxable_social_security_benefits: float = Field(description="Form 1040 Line 6b")
    capital_gain_or_loss: float = Field(description="Form 1040 Line 7")
    additional_income: float = Field(description="Form 1040 Line 8")
    total_income: float = Field(description="Form 1040 Line 9")
    total_adjustments: float = Field(description="Form 1040 Line 10")
    adjusted_gross_income: float = Field(description="Form 1040 Line 11")
    standard_or_itemized_deductions: float = Field(description="Form 1040 Line 12")
    qualified_business_income: float = Field(description="Form 1040 Line 13")
    total_deductions: float = Field(description="Form 1040 Line 14")
    computed_taxable_income: float = Field(description="Form 1040 Line 15")
    taxes: float = Field(description="Form 1040 Line 16")
    copy_schedule_2_line_3: float = Field(description="Form 1040 Line 17")
    f1040_line_18: float = Field(description="Form 1040 Line 18")
    ctc_or_other_dependent_credit: float = Field(description="Form 1040 Line 19")
    copy_schedule_3_line_8: float = Field(description="Form 1040 Line 20")
    accumulated_credits: float = Field(description="Form 1040 Line 21")
    taxes_after_credits: float = Field(description="Form 1040 Line 22")
    other_taxes: float = Field(description="Form 1040 Line 23")
    total_tax: float = Field(description="Form 1040 Line 24")
    federal_income_tax_withheld: float = Field(description="Form 1040 Line 25")
    earned_income_credit: float = Field(description="Form 1040 Line 27")
    additional_child_tax_credit: float = Field(description="Form 1040 Line 28")
    american_opportunity_credit: float = Field(description="Form 1040 Line 29")
    copy_schedule_3_line_15: float = Field(description="Form 1040 Line 31")
    total_other_payments_and_refundable_credits: float = Field(
        description="Form 1040 Line 32"
    )
    total_payments: float = Field(description="Form 1040 Line 33")
    amount_owed_or_overpaid: float = Field(
        description="Form 1040 Line 37 (negative if overpaid)"
    )
    taxable_state_refunds: float = Field(description="Schedule 1 Line 1")
    alimony_income: float = Field(description="Schedule 1 Line 2a")
    sale_of_business: float = Field(description="Schedule 1 Line 4")
    rental_real_estate_sch1: float = Field(description="Schedule 1 Line 5")
    farm_income: float = Field(description="Schedule 1 Line 6")
    unemployment_compensation: float = Field(description="Schedule 1 Line 7")
    other_income: float = Field(description="Schedule 1 Line 8")
    educator_expenses: float = Field(description="Schedule 1 Line 11")
    hsa_deduction: float = Field(description="Schedule 1 Line 13")
    self_employment_deductible: float = Field(description="Schedule 1 Line 15")
    ira_deduction: float = Field(description="Schedule 1 Line 20")
    student_loan_interest_deduction: float = Field(description="Schedule 1 Line 21")
    other_adjustments: float = Field(description="Schedule 1 Line 24")
    amt_f6251: float = Field(description="Schedule 2 Line 1")
    credit_repayment: float = Field(description="Schedule 2 Line 2")
    schedule_2_total_taxes: float = Field(
        description="Schedule 2 Line 3 (= Line 1 + Line 2)"
    )
    self_employment_tax: float = Field(description="Schedule 2 Line 4")
    other_additional_taxes: float = Field(description="Schedule 2 Line 17")
    schedule_2_total_other_taxes: float = Field(
        description="Schedule 2 Line 21 (= Line 4 + Line 17)"
    )
    foreign_tax_credit: float = Field(description="Schedule 3 Line 1")
    dependent_care: float = Field(description="Schedule 3 Line 2")
    computed_education_credits: float = Field(description="Schedule 3 Line 3")
    retirement_savings: float = Field(description="Schedule 3 Line 4")
    elderly_disabled_credits: float = Field(description="Schedule 3 Line 6d")
    plug_in_motor_vehicle: float = Field(description="Schedule 3 Line 6i")
    alt_motor_vehicle: float = Field(description="Schedule 3 Line 6j")
    schedule_3_line_8: float = Field(description="Schedule 3 Line 8")
    medical_dental_expenses: Optional[float] = Field(
        description="Schedule A Line 1 (if itemized)"
    )
    state_local_income_or_sales_tax: Optional[float] = Field(
        description="Schedule A Line 5a (if itemized)"
    )
    state_local_real_estate_tax: Optional[float] = Field(
        description="Schedule A Line 5b (if itemized)"
    )
    state_local_personal_property_tax: Optional[float] = Field(
        description="Schedule A Line 5c (if itemized)"
    )
    other_taxes_paid: Optional[float] = Field(
        description="Schedule A Line 6 (if itemized)"
    )
    home_mortgage_interest_and_points: Optional[float] = Field(
        description="Schedule A Line 8a (if itemized)"
    )
    home_mortgage_interest_unreported: Optional[float] = Field(
        description="Schedule A Line 8b"
    )
    home_mortgage_points_unreported: Optional[float] = Field(
        description="Schedule A Line 8c (if itemized)"
    )
    investment_interest: Optional[float] = Field(
        description="Schedule A Line 9 (if itemized)"
    )
    charity_cash: Optional[float] = Field(
        description="Schedule A Line 11 (if itemized)"
    )
    charity_non_cash: Optional[float] = Field(
        description="Schedule A Line 12 (if itemized)"
    )
    casualty_and_theft_loss: Optional[float] = Field(
        description="Schedule A Line 15 (if itemized)"
    )
    other_itemized_deductions: Optional[float] = Field(
        description="Schedule A Line 16 (if itemized)"
    )
    gross_receipts: Optional[float] = Field(
        description="Schedule C Line 1 (if self-employed)"
    )
    returns_and_allowances: Optional[float] = Field(
        description="Schedule C Line 2 (if self-employed)"
    )
    cost_of_goods_sold: Optional[float] = Field(
        description="Schedule C Line 4 (if self-employed)"
    )
    other_inc_sched_c: Optional[float] = Field(
        description="Schedule C Line 6 (if self-employed)"
    )
    total_expenses: Optional[float] = Field(
        description="Schedule C Line 28 (if self-employed)"
    )
    expenses_of_home: Optional[float] = Field(
        description="Schedule C Line 30 (if self-employed)"
    )
    net_profit: Optional[float] = Field(
        description="Schedule C Line 31 (if self-employed)"
    )
    total_social_security_wages: Optional[float] = Field(
        description="Schedule SE Line 8 (if self-employed)"
    )
    student_list: Optional[list[Student]] = Field(
        description="List of students with education expenses"
    )


class TaxPayer(BaseModel):
    name: str = Field(description="Name of taxpayer")
    age: int = Field(description="Age of taxpayer")
    spouse_age: int = Field(description="Age of taxpayer's spouse")
    filing_status: FilingStatus = Field(description="Filing status of taxpayer")
    blind: bool = Field(description="Taxpayer is blind")
    spouse_blind: bool = Field(description="Taxpayer's spouse is blind")
    itemized: bool = Field(description="Taxpayer uses itemized deductions")
    self_employed: bool = Field(description="Taxpayer is self-employed")
    has_student_loans_or_education_expenses: bool = Field(
        description="Taxpayer has student loans or education expenses"
    )
    num_qualifying_children: int = Field(description="Number of qualifying children")
    num_other_dependents: int = Field(description="Number of other dependents")
    wage_tip_compensation: float = Field(description="Form 1040 Line 1a")
    household_employee_wage: float = Field(description="Form 1040 Line 1b")
    unreported_tip: float = Field(description="Form 1040 Line 1c")
    nontaxable_combat_pay: float = Field(description="Form 1040 Line 1d")
    tax_exempt_interest: float = Field(description="Form 1040 Line 2a")
    taxable_interest: float = Field(description="Form 1040 Line 2b")
    qualified_dividends: float = Field(description="Form 1040 Line 3a")
    ordinary_dividends: float = Field(description="Form 1040 Line 3b")
    ira_distributions: float = Field(description="Form 1040 Line 4a")
    taxable_ira_distributions: float = Field(description="Form 1040 Line 4b")
    all_pensions: float = Field(description="Form 1040 Line 5a")
    taxable_pensions: float = Field(description="Form 1040 Line 5b")
    social_security_benefits: float = Field(description="Form 1040 Line 6a")
    taxable_social_security_benefits: float = Field(description="Form 1040 Line 6b")
    qualified_business_income: float = Field(description="Form 1040 Line 13")
    federal_income_tax_withheld: float = Field(description="Form 1040 Line 25")
    earned_income_credit: float = Field(description="Form 1040 Line 27")
    taxable_state_refunds: float = Field(description="Schedule 1 Line 1")
    alimony_income: float = Field(description="Schedule 1 Line 2a")
    sale_of_business: float = Field(description="Schedule 1 Line 4")
    rental_real_estate_sch1: float = Field(description="Schedule 1 Line 5")
    farm_income: float = Field(description="Schedule 1 Line 6")
    unemployment_compensation: float = Field(description="Schedule 1 Line 7")
    other_income: float = Field(description="Schedule 1 Line 8")
    educator_expenses: float = Field(description="Schedule 1 Line 11")
    hsa_deduction: float = Field(description="Schedule 1 Line 13")
    ira_deduction: float = Field(description="Schedule 1 Line 20")
    student_loan_interest_deduction: float = Field(description="Schedule 1 Line 21")
    other_adjustments: float = Field(description="Schedule 1 Line 24")
    amt_f6251: float = Field(description="Schedule 2 Line 1")
    credit_repayment: float = Field(description="Schedule 2 Line 2")
    other_additional_taxes: float = Field(description="Schedule 2 Line 17")
    foreign_tax_credit: float = Field(description="Schedule 3 Line 1")
    dependent_care: float = Field(description="Schedule 3 Line 2")
    retirement_savings: float = Field(description="Schedule 3 Line 4")
    elderly_disabled_credits: float = Field(description="Schedule 3 Line 6d")
    plug_in_motor_vehicle: float = Field(description="Schedule 3 Line 6i")
    alt_motor_vehicle: float = Field(description="Schedule 3 Line 6j")

    class Config:
        extra = "allow"
