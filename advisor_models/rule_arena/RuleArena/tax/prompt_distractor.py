basic_forms_distractor = """
# Basic Information

Name: $name
Age (on January 2, 2024): $age
Age (on January 2, 2024) of Your Spouse: $spouse_age

# Form 1040: U.S. Individual Income Tax Return

Filing Status: $filing_status
Using Itemized Deductions: $itemized

Age/Blindness:
* You were born before January 2, 1959: $TBD
* You were blind: $blind
* You spouse was born before January 2, 1959: $TBD
* You spouse was blind: $spouse_blind

## Dependents

Qualifying Children: $num_qualifying_children

Other Dependents: $num_other_dependents

## Income

Line 1a - Total amount from Form(s) W-2, box 1: $wage_tip_compensation

Line 1b - Household employee wages not reported on Form(s) W-2: $household_employee_wage

Line 1c - Tip income not reported on line 1a: $unreported_tip

Line 1d - Nontaxable combat pay election: $nontaxable_combat_pay

Line 1z - Add lines 1a through 1c: $TBD

Line 2a - Tax-exempt interest: $tax_exempt_interest

Line 2b - Taxable interest: $taxable_interest

Line 3a - Qualified dividends: $qualified_dividends

Line 3b - Ordinary dividends: $ordinary_dividends

Line 4a - IRA distributions: $ira_distributions

Line 4b - Taxable amount: $taxable_ira_distributions

Line 5a - Pensions and annuities: $all_pensions

Line 5b - Taxable amount: $taxable_pensions

Line 6a - Social security benefits: $social_security_benefits

Line 6b - Taxable amount: $taxable_social_security_benefits

Line 7 - Capital gain or (loss): $0

Line 8 - Additional income from Schedule 1, line 10: $TBD

Line 9 - Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income: $TBD

Line 10 - Adjustments to income from Schedule 1, line 26: $TBD

Line 11 - Subtract line 10 from line 9. This is your adjusted gross income: $TBD

### Instruction for Line 12:
Standard Deduction for—
* Single or Married filing separately, $13,850
* Married filing jointly or Qualifying surviving spouse, $27,700
* Head of household, $20,800
* If you checked any box under Standard Deduction, see instructions.

Most Form 1040 filers can find their standard deduction by looking at the amounts listed above. Most Form 1040-SR filers can find their standard deduction by using the chart on the last page of Form 1040-SR.

Exception—Born before January 2, 1959, or blind. If you checked any of the following boxes, figure your standard deduction using the Standard Deduction Chart for People Who Were Born Before January 2, 1959, or Were Blind if you are filing Form 1040 or by using the chart on the last page of Form 1040-SR.
* You were born before January 2, 1959.
* You are blind.
* Spouse was born before January 2, 1959.
* Spouse is blind.

You can only check the boxes for your spouse if your spouse is age 65 or older or blind and you file a joint return.

#### Standard Deduction Chart for People Who Were Born Before January 2, 1959, or Were Blind

According to the number of following boxes checked:
* You were born before January 2, 1959.
* You are blind.
* Spouse was born before January 2, 1959.
* Spouse is blind.

| IF your filing status is ... | AND the number in the box above is ... | THEN your standard deduction is ... |
| ---------------------------- | ----- | --------- |
| Single                       | 1     | $15,700   |
| Single                       | 2     | $17,550   |
| Married filing jointly       | 1     | $29,200   |
| Married filing jointly       | 2     | $30,700   |
| Married filing jointly       | 3     | $32,200   |
| Married filing jointly       | 4     | $33,700   |
| Qualifying surviving spouse  | 1     | $29,200   |
| Qualifying surviving spouse  | 2     | $30,700   |
| Married filing separately    | 1     | $15,350   |
| Married filing separately    | 2     | $16,850   |
| Head of household            | 1     | $22,650   |
| Head of household            | 2     | $24,500   |

Line 12 - Standard deduction or itemized deductions (from Schedule A): $TBD

Line 13 - Qualified business income deduction from Form 8995 or Form 8995-A: $qualified_business_income

Line 14 - Add lines 12 and 13: $TBD

Line 15 - Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income: $TBD

## Tax and Credits

### Instruction for Line 16

If your taxable income is less than $100,000, you must use the Tax Table, later in these instructions, to figure your tax. Be sure you use the correct column. If your taxable income is $100,000 or more, use the Tax Computation Worksheet right after the Tax Table.

#### 2023 Tax Table

First divide your taxable income by 50, round it down to the nearest integer, then multiply it by 50, and finally add 25. Enter the result (called rounded taxable income) here: $TBD

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | $TBD     | × 10% (0.10)     | $ 0           | $TBD     | $TBD     |
| Over $11,000 but not over $44,725  | $TBD     | × 12% (0.12)     | $ 220.00      | $TBD     | $TBD     |
| Over $44,725 but not over $95,375  | $TBD     | × 22% (0.22)     | $ 4692.50     | $TBD     | $TBD     |
| Over $95,375 but not over $100,000 | $TBD     | × 24% (0.24)     | $ 6600.00     | $TBD     | $TBD     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $22,000   | $TBD     | × 10% (0.10)     | $ 0           | $TBD     | $TBD     |
| Over $22,000 but not over $89,450  | $TBD     | × 12% (0.12)     | $ 440.00      | $TBD     | $TBD     |
| Over $89,450 but not over $100,000 | $TBD     | × 22% (0.22)     | $ 9385.00     | $TBD     | $TBD     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | $TBD     | × 10% (0.10)     | $ 0           | $TBD     | $TBD     |
| Over $11,000 but not over $44,725  | $TBD     | × 12% (0.12)     | $ 220.00      | $TBD     | $TBD     |
| Over $44,725 but not over $95,375  | $TBD     | × 22% (0.22)     | $ 4692.50     | $TBD     | $TBD     |
| Over $95,375 but not over $100,000 | $TBD     | × 24% (0.24)     | $ 6600.00     | $TBD     | $TBD     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $15,700   | $TBD     | × 10% (0.10)     | $ 0           | $TBD     | $TBD     |
| Over $15,700 but not over $59,850  | $TBD     | × 12% (0.12)     | $ 314.00      | $TBD     | $TBD     |
| Over $59,850 but not over $95,350  | $TBD     | × 22% (0.22)     | $ 6299.00     | $TBD     | $TBD     |
| Over $95,350 but not over $100,000 | $TBD     | × 24% (0.24)     | $ 8206.00     | $TBD     | $TBD     |

#### 2023 Tax Computation Worksheet—Line 16

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | $TBD     | × 24% (0.24)     | $ 6,600.00      | $TBD     | $TBD     |
| Over $182,100 but not over $231,250     | $TBD     | × 32% (0.32)     | $ 21,168.00     | $TBD     | $TBD     |
| Over $231,250 but not over $578,125     | $TBD     | × 35% (0.35)     | $ 28,105.50     | $TBD     | $TBD     |
| Over $578,125                           | $TBD     | × 37% (0.37)     | $ 39,668.00     | $TBD     | $TBD     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $190,750 | $TBD     | × 22% (0.22)     | $ 9,385.00      | $TBD     | $TBD     |
| Over $190,750 but not over $364,200     | $TBD     | × 24% (0.24)     | $ 13,200.00     | $TBD     | $TBD     |
| Over $364,200 but not over $462,500     | $TBD     | × 32% (0.32)     | $ 42,336.00     | $TBD     | $TBD     |
| Over $462,500 but not over $693,750     | $TBD     | × 35% (0.35)     | $ 56,211.00     | $TBD     | $TBD     |
| Over $693,750                           | $TBD     | × 37% (0.37)     | $ 70086.00      | $TBD     | $TBD     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | $TBD     | × 24% (0.24)     | $ 6,600.00      | $TBD     | $TBD     |
| Over $182,100 but not over $231,250     | $TBD     | × 32% (0.32)     | $ 21,168.00     | $TBD     | $TBD     |
| Over $231,250 but not over $346,875     | $TBD     | × 35% (0.35)     | $ 28,105.50     | $TBD     | $TBD     |
| Over $346,875                           | $TBD     | × 37% (0.37)     | $ 35,043.00     | $TBD     | $TBD     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | $TBD     | × 24% (0.24)     | $ 8,206.00      | $TBD     | $TBD     |
| Over $182,100 but not over $231,250     | $TBD     | × 32% (0.32)     | $ 22,774.00     | $TBD     | $TBD     |
| Over $231,250 but not over $578,125     | $TBD     | × 35% (0.35)     | $ 29,711.50     | $TBD     | $TBD     |
| Over $578,125                           | $TBD     | × 37% (0.37)     | $ 41,273.50     | $TBD     | $TBD     |

However, don’t use the Tax Table or Tax Computation Worksheet to figure your tax if the following applies.

Qualified Dividends and Capital Gain Tax Worksheet: Use the Qualified Dividends and Capital Gain Tax Worksheet, later, to figure your tax if you don’t have to use the Schedule D Tax Worksheet and if any of the following applies.
* You reported qualified dividends on Form 1040 or 1040-SR, line 3a.
* You don’t have to file Schedule D and you reported capital gain distributions on Form 1040 or 1040-SR, line 7.
* You are filing Schedule D, and Schedule D, lines 15 and 16, are both more than zero.

#### Qualified Dividends and Capital Gain Tax Worksheet—Line 16

Line 1 - Enter the amount from Form 1040 or 1040-SR, line 15: $TBD

Line 2 - Enter the amount from Form 1040 or 1040-SR, line 3a: $TBD

Line 3 - Are you filing Schedule D? If Yes, enter the smaller of line 15 or line 16 of Schedule D. If either line 15 or line 16 is blank or a loss, enter -0-. If No, enter the amount from Form 1040 or 1040-SR, line 7: $TBD

Line 4 - Add lines 2 and 3: $TBD

Line 5 - Subtract line 4 from line 1. If zero or less, enter -0-: $TBD

Line 6 - Enter: $44,625 if single or married filing separately, $89,250 if married filing jointly or qualifying surviving spouse, $59,750 if head of household: $TBD

Line 7 - Enter the smaller of line 1 or line 6: $TBD

Line 8 - Enter the smaller of line 5 or line 7: $TBD

Line 9 - Subtract line 8 from line 7. This amount is taxed at 0%: $TBD

Line 10 - Enter the smaller of line 1 or line 4: $TBD

Line 11 - Enter the amount from line 9: $TBD

Line 12 - Subtract line 11 from line 10: $TBD

Line 13 - Enter: $492,300 if single, $276,900 if married filing separately, $553,850 if married filing jointly or qualifying surviving spouse, $523,050 if head of household: $TBD

Line 14 - Enter the smaller of line 1 or line 13: $TBD

Line 15 - Add lines 5 and 9: $TBD

Line 16 - Subtract line 15 from line 14. If zero or less, enter -0-: $TBD

Line 17 - Enter the smaller of line 12 or line 16: $TBD

Line 18 - Multiply line 17 by 15% (0.15): $TBD

Line 19 - Add lines 9 and 17: $TBD

Line 20 - Subtract line 19 from line 10: $TBD

Line 21 - Multiply line 20 by 20% (0.20): $TBD

Line 22 - Figure the tax on the amount on line 5. If the amount on line 5 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 5 is $100,000 or more, use the Tax Computation Worksheet: $TBD

Line 23 - Add lines 18, 21, and 22: $TBD

Line 24 - Figure the tax on the amount on line 1. If the amount on line 1 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 1 is $100,000 or more, use the Tax Computation Worksheet: $TBD

Line 25 - Tax on all taxable income. Enter the smaller of line 23 or line 24. Also include this amount on the entry space on Form 1040 or 1040-SR, line 16: $TBD

---

Line 16 - Tax (See instructions): $TBD

Line 17 - Amount from Schedule 2, line 3: $TBD

Line 18 - Add lines 16 and 17: $TBD

Line 19 - Child tax credit or credit for other dependents from Schedule 8812: $ctc_or_other_dependent_credit

Line 20 - Amount from Schedule 3, line 8: $TBD

Line 21 - Add lines 19 and 20: $TBD

Line 22 - Subtract line 21 from line 18. If zero or less, enter -0-: $TBD

Line 23 - Other taxes, including self-employment tax, from Schedule 2, line 21: $TBD

Line 24 - Add lines 22 and 23. This is your total tax: $TBD

## Payments

Line 25 - Federal income tax withheld: $federal_income_tax_withheld

Line 26 - 2023 estimated tax payments and amount applied from 2022 return: $0

Line 27 - Earned income credit (EIC): $earned_income_credit

Line 28 - Additional child tax credit from Schedule 8812: $additional_child_tax_credit

Line 29 - American opportunity credit from Form 8863, line 8: $american_opportunity_credit

Line 30 - Reserved for future use: $0

Line 31 - Amount from Schedule 3, line 15: $TBD

Line 32 - Add lines 27, 28, 29, and 31. These are your total other payments and refundable credits: $TBD

Line 33 - Add lines 25, 26, and 32. These are your total payments: $TBD

## Amount You Owe

Line 37 - Subtract line 33 from line 24. This is the amount you owe: $TBD
    
# Schedule 1 (Form 1040): Additional Income and Adjustments to Income

## Part I: Additional Income

Line 1 - Taxable refunds, credits, or offsets of state and local income taxes: $taxable_state_refunds

Line 2a - Alimony received: $alimony_income

Line 3 - Business income or (loss). Attach Schedule C: $TBD

Line 4 - Other gains or (losses): $sale_of_business

Line 5 - Rental real estate, royalties, partnerships, S corporations, trusts, etc. Attach Schedule E: $rental_real_estate_sch1

Line 6 - Farm income or (loss). Attach Schedule F: $farm_income

Line 7 - Unemployment compensation: $unemployment_compensation

Line 8 - Other income: $other_income

Line 10 - Combine lines 1 through 8. This is your additional income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 8: $TBD

## Part II: Adjustments to Income

Line 11 - Educator expenses: $educator_expenses

Line 12 - Certain business expenses of reservists, performing artists, and fee-basis government officials: $0

Line 13 - Health savings account deduction: $hsa_deduction

Line 14 - Moving expenses for members of the Armed Forces: $0

Line 15 - Deductible part of self-employment tax: $self_employment_deductible

Line 16 - Self-employed SEP, SIMPLE, and qualified plans: $0

Line 17 - Self-employed health insurance deduction: $0

Line 18 - Penalty on early withdrawal of savings: $0 

Line 19 - Alimony paid: $0 

Line 20 - IRA deduction: $ira_deduction

Line 21 - Student loan interest deduction: $student_loan_interest_deduction

Line 22 - Reserved for future use: $0

Line 23 - Archer MSA deduction: $0

Line 24 - Other adjustments: $other_adjustments

Line 26 - Add lines 11 through 24. These are your adjustments to income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 10: $TBD

# Schedule 2 (Form 1040): Additional Taxes

## Part I: Tax

Line 1 - Alternative minimum tax: $amt_f6251

Line 2 - Excess advance premium tax credit repayment. Attach Form 8962: $credit_repayment

Line 3 - Add lines 1 and 2. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 17: $TBD

## Part II: Other Taxes

Line 4 - Self-employment tax. Attach Schedule SE: $TBD

Line 17 - Other additional taxes: $other_additional_taxes

Line 21 - Add lines 4 and 17. These are your total other taxes. Enter here and on Form 1040 or 1040-SR, line 23, or Form 1040-NR, line 23b: $TBD

# Schedule 3 (Form 1040): Additional Credits and Payments

# Part I: Nonrefundable Credits

Line 1 - Foreign tax credit: $foreign_tax_credit

Line 2 - Credit for child and dependent care expenses from Form 2441, line 11: $dependent_care

Line 3 - Education credits from Form 8863, line 19: $education_credits

Line 4 - Retirement savings contributions credit: $retirement_savings

Line 5a - Residential clean energy credit from Form 5695, line 15: $0

Line 5b - Energy efficient home improvement credit from Form 5695, line 32: $0

Line 6 - Other nonrefundable credits:

Line 6d - Elderly or disabled credit from Schedule R: $elderly_disabled_credits

Line 6i - Qualified plug-in motor vehicle credit: $plug_in_motor_vehicle

Line 6j - Alternative fuel vehicle refueling property credit: $alt_motor_vehicle

Line 7 - Total other nonrefundable credits. Add lines 6d, 6i, and 6j: $TBD

Line 8 - Add lines 1 through 4, 5a, 5b, and 7. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 20: $TBD

# Schedule A (Form 1040): Itemized Deductions

## Medical and Dental Expenses

Line 1 - Medical and dental expenses: $0

Line 2 - Enter amount from Form 1040 or 1040-SR, line 11: $TBD

Line 3 - Multiply line 2 by 7.5% (0.075): $TBD

Line 4 - Subtract line 3 from line 1. If line 3 is more than line 1, enter -0-: $TBD

## Taxes You Paid

Line 5a - State and local income taxes or general sales taxes: $0

Line 5b - State and local real estate taxes: $0

Line 5c - State and local personal property taxes: $0

Line 5d - Add lines 5a through 5c: $TBD

Line 5e -  Enter the smaller of line 5d or $10,000 ($5,000 if married filing separately): $TBD

Line 6 - Other taxes: $0

Line 7 - Add lines 5e and 6: $TBD

## Interest You Paid

Line 8a - Home mortgage interest and points reported to you on Form 1098: $0

Line 8b - Home mortgage interest not reported to you on Form 1098: $0

Line 8c - Points not reported to you on Form 1098: $0

Line 8e - Add lines 8a through 8c: $TBD

Line 9 - Investment interest: $0

Line 10 - Add lines 8e and 9: $TBD

## Gifts to Charity

Line 11 - Gifts by cash or check: $0

Line 12 - Other than by cash or check: $0

Line 13 - Carryover from prior year: $0

Line 14 - Add lines 11 through 13: $TBD

## Casualty and Theft Losses

Line 15 - Casualty and theft loss(es) from a federally declared disaster (other than net qualified disaster losses): $0

## Other Itemized Deductions

Line 16 - Other: $0

## Total Itemized Deductions

Line 17 - Add the amounts in the far right column for lines 4 through 16. Also, enter this amount on Form 1040 or 1040-SR, line 12: $TBD

# Schedule C (Form 1040): Profit or Loss From Business

## Part I: Income

Line 1 - Gross receipts or sales: $0

Line 2 - Returns and allowances: $0

Line 3 - Subtract line 2 from line 1: $TBD

Line 4 - Cost of goods sold: $0

Line 5 - Gross profit. Subtract line 4 from line 3: $TBD

Line 6 - Other income, including federal and state gasoline or fuel tax credit or refund: $0

Line 7 - Gross income. Add lines 5 and 6: $TBD

## Part II: Expenses

Line 28 - Total expenses before expenses for business use of home: $0

Line 29 - Tentative profit or (loss). Subtract line 28 from line 7: $TBD

Line 30 - Expenses for business use of your home: $0

Line 31 - Net profit or (loss). Subtract line 30 from line 29. Enter here and on both Schedule 1 (Form 1040), line 3, and on Schedule SE, line 2: $TBD

# Schedule SE (Form 1040): Self-Employment Tax

## Part I: Self-Employment Tax

Line 3 - Net profit or (loss) from Schedule C, line 31; and Schedule K-1 (Form 1065), box 14, code A (other than farming): $TBD

Line 4c - If line 3 is more than zero, multiply line 3 by 92.35% (0.9235). Otherwise, enter amount from line 3: $TBD

Line 7 - Maximum amount of combined wages and self-employment earnings subject to social security tax or the 6.2% portion of the 7.65% railroad retirement (tier 1) tax for 2023: $160,200

Line 8 - Total social security wages and tips (total of boxes 3 and 7 on Form(s) W-2) and railroad retirement (tier 1) compensation. If $160,200 or more, skip lines 9 through 10, and go to line 11: $0

Line 9 - Subtract line 8 from line 7. If zero or less, enter -0- here and on line 10 and go to line 11: $TBD

Line 10 - Multiply the smaller of line 4 or line 9 by 12.4% (0.124): $TBD

Line 11 - Multiply line 4 by 2.9% (0.029): $TBD

Line 12 - Self-employment tax. Add lines 10 and 11. Enter here and on Schedule 2 (Form 1040), line 4, or Form 1040-SS, Part I, line 3: $TBD

Line 13 - Deduction for one-half of self-employment tax. Multiply line 12 by 50% (0.50). Enter here and on Schedule 1 (Form 1040), line 15: $TBD

# Form 8863: Education Credits (American Opportunity and Lifetime Learning Credits)

## Part I: Refundable American Opportunity Credit

Line 1 - After completing Part III for each student, enter the total of all amounts from all Parts III, line 30: $TBD

Line 2 - Enter: $180,000 if married filing jointly; $90,000 otherwise: $TBD

Line 3 - Enter the amount from Form 1040 or 1040-SR, line 11: $TBD

Line 4 - Subtract line 3 from line 2. If zero or less, stop; you can’t take any education credit: $TBD

Line 5 - Enter: $20,000 if married filing jointly; $10,000 otherwise: $TBD

Line 6 - If line 4 is: a) Equal to or more than line 5, enter 1.000 on line 6; b) Less than line 5, divide line 4 by line 5. Enter the result as a decimal (rounded to three places): $TBD

Line 7 - Multiply line 1 by line 6. Caution: If you were under age 24 at the end of the year and meet the conditions described in the instructions, you can’t take the refundable American opportunity credit; skip line 8, enter the amount from line 7 on line 9: $TBD

Line 8 - Refundable American opportunity credit. Multiply line 7 by 40% (0.40). Enter the amount here and on Form 1040 or 1040-SR, line 29. Then go to line 9 below: $TBD

## Part II: Nonrefundable Education Credits

Line 9 - Subtract line 8 from line 7. Enter here and on line 2 of the Credit Limit Worksheet (see instructions): $TBD

Line 10 - After completing Part III for each student, enter the total of all amounts from all Parts III, line 31. If zero, skip lines 11 through 17, enter -0- on line 18, and go to line 19: $TBD

Line 11 - Enter the smaller of line 10 or $10,000: $TBD

Line 12 - Multiply line 11 by 20% (0.20): $TBD

Line 13 - Enter: $180,000 if married filing jointly; $90,000 otherwise: $TBD

Line 14 - Enter the amount from Form 1040 or 1040-SR, line 11: $TBD

Line 15 - Subtract line 14 from line 13. If zero or less, skip lines 16 and 17, enter -0- on line 18, and go to line 19: $TBD

Line 16 - Enter: $20,000 if married filing jointly; $10,000 otherwise: $TBD

Line 17 - If line 15 is: a) Equal to or more than line 16, enter 1.000 on line 17 and go to line 18; b) Less than line 16, divide line 15 by line 16. Enter the result as a decimal (rounded to three places): $TBD

Line 18 - Multiply line 12 by line 17. Enter here and on line 1 of the Credit Limit Worksheet (see instructions): $TBD

### Credit Limit Worksheet - Complete this worksheet to figure the amount to enter on line 19.

1. Enter the amount from Form 8863, line 18: $TBD

2. Enter the amount from Form 8863, line 9: $TBD

3. Add lines 1 and 2: $TBD

4. Enter the amount from Form 1040 or 1040-SR, line 18: $TBD

5. Enter the total of your credits from Schedule 3 (Form 1040), lines 1, 2, 6d, and 6l: $TBD

6. Subtract line 5 from line 4: $TBD

7. Enter the smaller of line 3 or line 6 here and on Form 8863, line 19: $TBD

Line 19 - Nonrefundable education credits. Enter the amount from line 7 of the Credit Limit Worksheet (see instructions) here and on Schedule 3 (Form 1040), line 3: $TBD

## Part III: Student and Educational Institution Information. See instructions.

Line 21 - Adjusted qualified education expenses. Enter on Line 27 or Line 31: $0

Line 23 - Has the American opportunity credit been claimed for this student for any 4 prior tax years?: No
* Yes - Stop! Go to line 31 for this student.
* No - Go to line 24.

Line 24 - Was the student enrolled at least half-time for at least one academic period that began or is treated as having begun in 2024 at an eligible educational institution in a program leading towards a postsecondary degree, certificate, or other recognized postsecondary educational credential?: Yes
* Yes - Go to line 25.
* No - Stop! Go to line 31 for this student.

Line 25 - Did the student complete the first 4 years of postsecondary education before 2024?: No
* Yes - Stop! Go to line 31 for this student.
* No - Go to line 26.

Line 26 - Was the student convicted, before the end of 2024, of a felony for possession or distribution of a controlled substance?: No
* Yes - Stop! Go to line 31 for this student.
* No - Complete lines 27 through 30 for this student.

### American Opportunity Credit

Line 27 - Adjusted qualified education expenses (see instructions). Don’t enter more than $4,000: $TBD

Line 28 - Subtract $2,000 from line 27. If zero or less, enter -0-: $TBD

Line 29 - Multiply line 28 by 25% (0.25): $TBD

Line 30 - If line 28 is zero, enter the amount from line 27. Otherwise, add $2,000 to the amount on line 29 and enter the result. Skip line 31. Include the total of all amounts from all Parts III, line 30, on Part I, line 1: $TBD

### Lifetime Learning Credit

Line 31 - Adjusted qualified education expenses (see instructions). Include the total of all amounts from all Parts III, line 31, on Part II, line 10: $TBD

# Schedule 8812: Credits for Qualifying Children and Other Dependents

## Part I: Child Tax Credit and Credit for Other Dependents

Line 1 - Enter the amount from line 11 of your Form 1040, 1040-SR, or 1040-NR: $TBD

Line 4 - Number of qualifying children under age 17: $TBD

Line 5 - Multiply line 4 by $2,000: $TBD

Line 6 - Number of other dependents, including any qualifying children who are not under age 17: $TBD
Caution: Do not include yourself, your spouse, or anyone who is not a U.S. citizen, U.S. national, or U.S. resident alien. Also, do not include anyone you included on line 4.

Line 7 - Multiply line 6 by $500: $TBD

Line 8 - Add lines 5 and 7: $TBD

Line 9 - Enter the amount shown below for your filing status: Married filing jointly—$400,000; All other filing statuses—$200,000: $TBD

Line 10 - Subtract line 9 from line 1: $TBD
* If zero or less, enter -0-;
* If more than zero and not a multiple of $1,000, enter the next multiple of $1,000. For example, if the result is $425, enter $1,000; if the result is $1,025, enter $2,000, etc.

Line 11 - Multiply line 10 by 5% (0.05): $TBD

Line 12 - Is the amount on line 8 more than the amount on line 11?: $TBD
* No. STOP. You cannot take the child tax credit, credit for other dependents, or additional child tax credit. Skip Parts II-A and II-B. Enter -0- on lines 14 and 27.
* Yes. Subtract line 11 from line 8. Enter the result.

### Credit Limit Worksheet A

1. Enter the amount from line 18 of your Form 1040, 1040–SR, or 1040–NR: $TBD

2. Add the following amounts (if applicable) from:

2.1 Schedule 3, line 1: $TBD

2.2 Schedule 3, line 2: $TBD

2.3 Schedule 3, line 3: $TBD

2.4 Schedule 3, line 4: $TBD

2.5 Schedule 3, line 5b: $TBD

2.6 Schedule 3, line 6d: $TBD

2.7 Schedule 3, line 6f: $TBD

2.8 Schedule 3, line 6l: $TBD

2.9 Schedule 3, line 6m: $TBD

3. Subtract line 2 from line 1: $TBD

Complete Credit Limit Worksheet B only if Line 4 of Schedule 8812 is more than zero.

4. If you are not completing Credit Limit Worksheet B, enter -0-; otherwise, enter the amount from Credit Limit Worksheet B: $TBD

5. Subtract line 4 from line 3. Enter here and on Schedule 8812, line 13.

### Credit Limit Worksheet B

1. Enter the amount from Schedule 8812, line 12: $TBD

2. Number of qualifying children under 17 with the required social security number × $1,600. Enter the result: $TBD

TIP: The number of children you use for this line is the same as the number of children you used for line 4 of Schedule 8812.

3. Enter your earned income from line 7 of the Earned Income Worksheet: $TBD

4. Is the amount on line 3 more than $2,500?: $TBD
* No. Leave line 4 blank, enter -0- on line 5, and go to line 6.
* Yes. Subtract $2,500 from the amount on line 3. Enter the result

5. Multiply the amount on line 4 by 15% (0.15) and enter the result: $TBD

6. On line 2 of this worksheet, is the amount $4,800 or more?: $TBD
* No. Leave line 7 through 10 blank, enter -0- on line 11, and go to line 12.
* Yes. If line 5 above is equal to or more than line 1 above, leave lines 7 through 10 blank, enter -0- on line 11, and go to line 12. Otherwise, go to line 7.

7. Social security tax withheld from Form(s) W-2, box 4: $0

8. Enter the total of any amounts from — Schedule 1, line 15; Schedule 2, line 5; Schedule 2, line 6; and Schedule 2, line 13: $TBD

9. Add lines 7 and 8. Enter the total: $TBD

10. Enter the amounts from Form 1040 or 1040-SR, line 27: $TBD

11. Subtract line 10 from line 9. If the result is zero or less, enter -0-: $TBD

12. Enter the larger of line 5 or line 11: $TBD

13. Enter the smaller of line 2 or line 12: $TBD

14. Is the amount on line 13 of this worksheet more than the amount on line 1?: $TBD
* No. Subtract line 13 from line 1. Enter the result.
* Yes. Enter -0-.

15. Enter the total of the amounts from — Schedule 3, line 5a; Schedule 3, line 6c; Schedule 3, line 6g; and Schedule 3, line 6h: $TBD
Enter this amount on line 4 of the Credit Limit Worksheet A.

Line 13 - Enter the amount from Credit Limit Worksheet A: $TBD

Line 14 - Enter the smaller of line 12 or line 13. This is your child tax credit and credit for other dependents. Enter this amount on Form 1040, 1040-SR, or 1040-NR, line 19: $TBD

If the amount on line 12 is more than the amount on line 14, you may be able to take the additional child tax credit on Form 1040, 1040-SR, or 1040-NR, line 28. Complete your Form 1040, 1040-SR, or 1040-NR through line 27 (also complete Schedule 3, line 11) before completing Part II-A.

## Part II-A: Additional Child Tax Credit for All Filers

Line 16a - Subtract line 14 from line 12. If zero, stop here; you cannot take the additional child tax credit. Skip Parts II-A and II-B. Enter -0- on line 27: $TBD

Line 16b - Number of qualifying children under 17 x $1,600. Enter the result. If zero, stop here; you cannot claim the additional child tax credit. Skip Parts II-A and II-B. Enter -0- on line 27: $TBD
TIP: The number of children you use for this line is the same as the number of children you used for line 4.

Line 17 - Enter the smaller of line 16a or line 16b: $TBD

### Instruction for line 18a

Use the Earned Income Worksheet next to figure the amount to enter on line 18a.

#### Earned Income Worksheet

1. a. Enter the amount from line 1z of Form 1040, 1040-SR, or 1040-NR: $TBD

1. b. Enter the amount of any nontaxable combat pay received. Also enter this amount on Schedule 8812, line 18b. This amount will be reported on line 1d of Form 1040 or 1040-SR: $TBD

Next, if you are filing Schedule C, F, or SE, or you received a Schedule K-1 (Form 1065), go to line 2a. Otherwise, skip lines 2a through 2e and go to line 3.

2. a. Enter any statutory employee income reported on line 1 of Schedule C: $0

2. b. Enter any net profit or (loss) from Schedule C, line 31: $TBD

2. c. Enter any net farm profit or (loss) from Schedule F, line 34: $0

3. Combine lines 1a, 1b, 2a, 2b, and 2c. If zero or less, stop. Do not complete the rest of this worksheet. Instead, enter -0- on line 3 of Credit Limit Worksheet B or line 18a of Schedule 8812, whichever applies: $TBD

4. Enter the Medicaid waiver payment amounts excluded from income on Schedule 1 (Form 1040), line 8s: $0

5. Enter the amount from Schedule 1 (Form 1040), line 15: $TBD

6. Add lines 4 and 5: $TBD

7. Subtract line 6 from line 3, enter this amount on line 18a of Schedule 8812: $TBD

Line 18a - Earned income (see instructions): $TBD

Line 18b - Enter the amount of any nontaxable combat pay received: $TBD

Line 19 - Is the amount on line 18a more than $2,500?: $TBD
* No. Leave line 19 blank and enter -0- on line 20.
* Yes. Subtract $2,500 from the amount on line 18a. Enter the result.

Line 20 - Multiply the amount on line 19 by 15% (0.15) and enter the result: $TBD

Next. On line 16b, is the amount $4,800 or more?
* No. Skip Part II-B and enter the smaller of line 17 or line 20 on line 27.
* Yes. If line 20 is equal to or more than line 17, skip Part II-B and enter the amount from line 17 on line 27. Otherwise, go to line 22.

# Part II-B: Certain Filers Who Have Three or More Qualifying Children

Line 22 - Enter the total of the amounts from Schedule 1 (Form 1040), line 15; Schedule 2 (Form 1040), line 5; Schedule 2 (Form 1040), line 6; and Schedule 2 (Form 1040), line 13: $TBD

Line 24 - Enter the total of the amounts from Form 1040 or 1040-SR, line 27, and Schedule 3 (Form 1040), line 11: $TBD

Line 25 - Subtract line 24 from line 22. If zero or less, enter -0-: $TBD

Line 26 - Enter the larger of line 20 or line 25: $TBD

# Part II-C: Additional Child Tax Credit

Line 27 - Enter the smaller of line 17 or line 26 on line 27. This is your additional child tax credit. Enter this amount on Form 1040, 1040-SR, or 1040-NR, line 28: $TBD
"""

prompt_placeholder = """
# Basic Information

Name: $name
Age (on January 2, 2024): $age
Age (on January 2, 2024) of Your Spouse: $spouse_age

# Form 1040: U.S. Individual Income Tax Return

Filing Status: $filing_status
Using Itemized Deductions: $itemized

Age/Blindness:
* You were born before January 2, 1959: $TBD
* You were blind: $blind
* You spouse was born before January 2, 1959: $TBD
* You spouse was blind: $spouse_blind

## Dependents

Qualifying Children: $num_qualifying_children

Other Dependents: $num_other_dependents

## Income

Line 1a - Total amount from Form(s) W-2, box 1: $wage_tip_compensation

Line 1b - Household employee wages not reported on Form(s) W-2: $household_employee_wage

Line 1c - Tip income not reported on line 1a: $unreported_tip

Line 1d - Nontaxable combat pay election: $nontaxable_combat_pay

Line 1z - Add lines 1a through 1c: $TBD

Line 2a - Tax-exempt interest: $tax_exempt_interest

Line 2b - Taxable interest: $taxable_interest

Line 3a - Qualified dividends: $qualified_dividends

Line 3b - Ordinary dividends: $ordinary_dividends

Line 4a - IRA distributions: $ira_distributions

Line 4b - Taxable amount: $taxable_ira_distributions

Line 5a - Pensions and annuities: $all_pensions

Line 5b - Taxable amount: $taxable_pensions

Line 6a - Social security benefits: $social_security_benefits

Line 6b - Taxable amount: $taxable_social_security_benefits

Line 7 - Capital gain or (loss): $0

Line 8 - Additional income from Schedule 1, line 10: $TBD

Line 9 - Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income: $TBD

Line 10 - Adjustments to income from Schedule 1, line 26: $TBD

Line 11 - Subtract line 10 from line 9. This is your adjusted gross income: $TBD

### Instruction for Line 12:
Standard Deduction for—
* Single or Married filing separately, $13,850
* Married filing jointly or Qualifying surviving spouse, $27,700
* Head of household, $20,800
* If you checked any box under Standard Deduction, see instructions.

Most Form 1040 filers can find their standard deduction by looking at the amounts listed above. Most Form 1040-SR filers can find their standard deduction by using the chart on the last page of Form 1040-SR.

Exception—Born before January 2, 1959, or blind. If you checked any of the following boxes, figure your standard deduction using the Standard Deduction Chart for People Who Were Born Before January 2, 1959, or Were Blind if you are filing Form 1040 or by using the chart on the last page of Form 1040-SR.
* You were born before January 2, 1959.
* You are blind.
* Spouse was born before January 2, 1959.
* Spouse is blind.

You can only check the boxes for your spouse if your spouse is age 65 or older or blind and you file a joint return.

#### Standard Deduction Chart for People Who Were Born Before January 2, 1959, or Were Blind

According to the number of following boxes checked:
* You were born before January 2, 1959.
* You are blind.
* Spouse was born before January 2, 1959.
* Spouse is blind.

| IF your filing status is ... | AND the number in the box above is ... | THEN your standard deduction is ... |
| ---------------------------- | ----- | --------- |
| Single                       | 1     | $15,700   |
| Single                       | 2     | $17,550   |
| Married filing jointly       | 1     | $29,200   |
| Married filing jointly       | 2     | $30,700   |
| Married filing jointly       | 3     | $32,200   |
| Married filing jointly       | 4     | $33,700   |
| Qualifying surviving spouse  | 1     | $29,200   |
| Qualifying surviving spouse  | 2     | $30,700   |
| Married filing separately    | 1     | $15,350   |
| Married filing separately    | 2     | $16,850   |
| Head of household            | 1     | $22,650   |
| Head of household            | 2     | $24,500   |

Line 12 - Standard deduction or itemized deductions (from Schedule A): $TBD

Line 13 - Qualified business income deduction from Form 8995 or Form 8995-A: $qualified_business_income

Line 14 - Add lines 12 and 13: $TBD

Line 15 - Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income: $TBD

## Tax and Credits

### Instruction for Line 16

If your taxable income is less than $100,000, you must use the Tax Table, later in these instructions, to figure your tax. Be sure you use the correct column. If your taxable income is $100,000 or more, use the Tax Computation Worksheet right after the Tax Table.

#### 2023 Tax Table

First divide your taxable income by 50, round it down to the nearest integer, then multiply it by 50, and finally add 25. Enter the result (called rounded taxable income) here: $TBD

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | $TBD     | × 10% (0.10)     | $ 0           | $TBD     | $TBD     |
| Over $11,000 but not over $44,725  | $TBD     | × 12% (0.12)     | $ 220.00      | $TBD     | $TBD     |
| Over $44,725 but not over $95,375  | $TBD     | × 22% (0.22)     | $ 4692.50     | $TBD     | $TBD     |
| Over $95,375 but not over $100,000 | $TBD     | × 24% (0.24)     | $ 6600.00     | $TBD     | $TBD     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $22,000   | $TBD     | × 10% (0.10)     | $ 0           | $TBD     | $TBD     |
| Over $22,000 but not over $89,450  | $TBD     | × 12% (0.12)     | $ 440.00      | $TBD     | $TBD     |
| Over $89,450 but not over $100,000 | $TBD     | × 22% (0.22)     | $ 9385.00     | $TBD     | $TBD     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | $TBD     | × 10% (0.10)     | $ 0           | $TBD     | $TBD     |
| Over $11,000 but not over $44,725  | $TBD     | × 12% (0.12)     | $ 220.00      | $TBD     | $TBD     |
| Over $44,725 but not over $95,375  | $TBD     | × 22% (0.22)     | $ 4692.50     | $TBD     | $TBD     |
| Over $95,375 but not over $100,000 | $TBD     | × 24% (0.24)     | $ 6600.00     | $TBD     | $TBD     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $15,700   | $TBD     | × 10% (0.10)     | $ 0           | $TBD     | $TBD     |
| Over $15,700 but not over $59,850  | $TBD     | × 12% (0.12)     | $ 314.00      | $TBD     | $TBD     |
| Over $59,850 but not over $95,350  | $TBD     | × 22% (0.22)     | $ 6299.00     | $TBD     | $TBD     |
| Over $95,350 but not over $100,000 | $TBD     | × 24% (0.24)     | $ 8206.00     | $TBD     | $TBD     |

#### 2023 Tax Computation Worksheet—Line 16

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | $TBD     | × 24% (0.24)     | $ 6,600.00      | $TBD     | $TBD     |
| Over $182,100 but not over $231,250     | $TBD     | × 32% (0.32)     | $ 21,168.00     | $TBD     | $TBD     |
| Over $231,250 but not over $578,125     | $TBD     | × 35% (0.35)     | $ 28,105.50     | $TBD     | $TBD     |
| Over $578,125                           | $TBD     | × 37% (0.37)     | $ 39,668.00     | $TBD     | $TBD     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $190,750 | $TBD     | × 22% (0.22)     | $ 9,385.00      | $TBD     | $TBD     |
| Over $190,750 but not over $364,200     | $TBD     | × 24% (0.24)     | $ 13,200.00     | $TBD     | $TBD     |
| Over $364,200 but not over $462,500     | $TBD     | × 32% (0.32)     | $ 42,336.00     | $TBD     | $TBD     |
| Over $462,500 but not over $693,750     | $TBD     | × 35% (0.35)     | $ 56,211.00     | $TBD     | $TBD     |
| Over $693,750                           | $TBD     | × 37% (0.37)     | $ 70086.00      | $TBD     | $TBD     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | $TBD     | × 24% (0.24)     | $ 6,600.00      | $TBD     | $TBD     |
| Over $182,100 but not over $231,250     | $TBD     | × 32% (0.32)     | $ 21,168.00     | $TBD     | $TBD     |
| Over $231,250 but not over $346,875     | $TBD     | × 35% (0.35)     | $ 28,105.50     | $TBD     | $TBD     |
| Over $346,875                           | $TBD     | × 37% (0.37)     | $ 35,043.00     | $TBD     | $TBD     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | $TBD     | × 24% (0.24)     | $ 8,206.00      | $TBD     | $TBD     |
| Over $182,100 but not over $231,250     | $TBD     | × 32% (0.32)     | $ 22,774.00     | $TBD     | $TBD     |
| Over $231,250 but not over $578,125     | $TBD     | × 35% (0.35)     | $ 29,711.50     | $TBD     | $TBD     |
| Over $578,125                           | $TBD     | × 37% (0.37)     | $ 41,273.50     | $TBD     | $TBD     |

However, don’t use the Tax Table or Tax Computation Worksheet to figure your tax if the following applies.

Qualified Dividends and Capital Gain Tax Worksheet: Use the Qualified Dividends and Capital Gain Tax Worksheet, later, to figure your tax if you don’t have to use the Schedule D Tax Worksheet and if any of the following applies.
* You reported qualified dividends on Form 1040 or 1040-SR, line 3a.
* You don’t have to file Schedule D and you reported capital gain distributions on Form 1040 or 1040-SR, line 7.
* You are filing Schedule D, and Schedule D, lines 15 and 16, are both more than zero.

#### Qualified Dividends and Capital Gain Tax Worksheet—Line 16

Line 1 - Enter the amount from Form 1040 or 1040-SR, line 15: $TBD

Line 2 - Enter the amount from Form 1040 or 1040-SR, line 3a: $TBD

Line 3 - Are you filing Schedule D? If Yes, enter the smaller of line 15 or line 16 of Schedule D. If either line 15 or line 16 is blank or a loss, enter -0-. If No, enter the amount from Form 1040 or 1040-SR, line 7: $TBD

Line 4 - Add lines 2 and 3: $TBD

Line 5 - Subtract line 4 from line 1. If zero or less, enter -0-: $TBD

Line 6 - Enter: $44,625 if single or married filing separately, $89,250 if married filing jointly or qualifying surviving spouse, $59,750 if head of household: $TBD

Line 7 - Enter the smaller of line 1 or line 6: $TBD

Line 8 - Enter the smaller of line 5 or line 7: $TBD

Line 9 - Subtract line 8 from line 7. This amount is taxed at 0%: $TBD

Line 10 - Enter the smaller of line 1 or line 4: $TBD

Line 11 - Enter the amount from line 9: $TBD

Line 12 - Subtract line 11 from line 10: $TBD

Line 13 - Enter: $492,300 if single, $276,900 if married filing separately, $553,850 if married filing jointly or qualifying surviving spouse, $523,050 if head of household: $TBD

Line 14 - Enter the smaller of line 1 or line 13: $TBD

Line 15 - Add lines 5 and 9: $TBD

Line 16 - Subtract line 15 from line 14. If zero or less, enter -0-: $TBD

Line 17 - Enter the smaller of line 12 or line 16: $TBD

Line 18 - Multiply line 17 by 15% (0.15): $TBD

Line 19 - Add lines 9 and 17: $TBD

Line 20 - Subtract line 19 from line 10: $TBD

Line 21 - Multiply line 20 by 20% (0.20): $TBD

Line 22 - Figure the tax on the amount on line 5. If the amount on line 5 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 5 is $100,000 or more, use the Tax Computation Worksheet: $TBD

Line 23 - Add lines 18, 21, and 22: $TBD

Line 24 - Figure the tax on the amount on line 1. If the amount on line 1 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 1 is $100,000 or more, use the Tax Computation Worksheet: $TBD

Line 25 - Tax on all taxable income. Enter the smaller of line 23 or line 24. Also include this amount on the entry space on Form 1040 or 1040-SR, line 16: $TBD

---

Line 16 - Tax (See instructions): $TBD

Line 17 - Amount from Schedule 2, line 3: $TBD

Line 18 - Add lines 16 and 17: $TBD

Line 19 - Child tax credit or credit for other dependents from Schedule 8812: $ctc_or_other_dependent_credit

Line 20 - Amount from Schedule 3, line 8: $TBD

Line 21 - Add lines 19 and 20: $TBD

Line 22 - Subtract line 21 from line 18. If zero or less, enter -0-: $TBD

Line 23 - Other taxes, including self-employment tax, from Schedule 2, line 21: $TBD

Line 24 - Add lines 22 and 23. This is your total tax: $TBD

## Payments

Line 25 - Federal income tax withheld: $federal_income_tax_withheld

Line 26 - 2023 estimated tax payments and amount applied from 2022 return: $0

Line 27 - Earned income credit (EIC): $earned_income_credit

Line 28 - Additional child tax credit from Schedule 8812: $additional_child_tax_credit

Line 29 - American opportunity credit from Form 8863, line 8: $american_opportunity_credit

Line 30 - Reserved for future use: $0

Line 31 - Amount from Schedule 3, line 15: $TBD

Line 32 - Add lines 27, 28, 29, and 31. These are your total other payments and refundable credits: $TBD

Line 33 - Add lines 25, 26, and 32. These are your total payments: $TBD

## Amount You Owe

Line 37 - Subtract line 33 from line 24. This is the amount you owe: $TBD
    
# Schedule 1 (Form 1040): Additional Income and Adjustments to Income

## Part I: Additional Income

Line 1 - Taxable refunds, credits, or offsets of state and local income taxes: $taxable_state_refunds

Line 2a - Alimony received: $alimony_income

Line 3 - Business income or (loss). Attach Schedule C: $TBD

Line 4 - Other gains or (losses): $sale_of_business

Line 5 - Rental real estate, royalties, partnerships, S corporations, trusts, etc. Attach Schedule E: $rental_real_estate_sch1

Line 6 - Farm income or (loss). Attach Schedule F: $farm_income

Line 7 - Unemployment compensation: $unemployment_compensation

Line 8 - Other income: $other_income

Line 10 - Combine lines 1 through 8. This is your additional income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 8: $TBD

## Part II: Adjustments to Income

Line 11 - Educator expenses: $educator_expenses

Line 12 - Certain business expenses of reservists, performing artists, and fee-basis government officials: $0

Line 13 - Health savings account deduction: $hsa_deduction

Line 14 - Moving expenses for members of the Armed Forces: $0

Line 15 - Deductible part of self-employment tax: $self_employment_deductible

Line 16 - Self-employed SEP, SIMPLE, and qualified plans: $0

Line 17 - Self-employed health insurance deduction: $0

Line 18 - Penalty on early withdrawal of savings: $0 

Line 19 - Alimony paid: $0 

Line 20 - IRA deduction: $ira_deduction

Line 21 - Student loan interest deduction: $student_loan_interest_deduction

Line 22 - Reserved for future use: $0

Line 23 - Archer MSA deduction: $0

Line 24 - Other adjustments: $other_adjustments

Line 26 - Add lines 11 through 24. These are your adjustments to income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 10: $TBD

# Schedule 2 (Form 1040): Additional Taxes

## Part I: Tax

Line 1 - Alternative minimum tax: $amt_f6251

Line 2 - Excess advance premium tax credit repayment. Attach Form 8962: $credit_repayment

Line 3 - Add lines 1 and 2. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 17: $TBD

## Part II: Other Taxes

Line 4 - Self-employment tax. Attach Schedule SE: $TBD

Line 17 - Other additional taxes: $other_additional_taxes

Line 21 - Add lines 4 and 17. These are your total other taxes. Enter here and on Form 1040 or 1040-SR, line 23, or Form 1040-NR, line 23b: $TBD

# Schedule 3 (Form 1040): Additional Credits and Payments

# Part I: Nonrefundable Credits

Line 1 - Foreign tax credit: $foreign_tax_credit

Line 2 - Credit for child and dependent care expenses from Form 2441, line 11: $dependent_care

Line 3 - Education credits from Form 8863, line 19: $education_credits

Line 4 - Retirement savings contributions credit: $retirement_savings

Line 5a - Residential clean energy credit from Form 5695, line 15: $0

Line 5b - Energy efficient home improvement credit from Form 5695, line 32: $0

Line 6 - Other nonrefundable credits:

Line 6d - Elderly or disabled credit from Schedule R: $elderly_disabled_credits

Line 6i - Qualified plug-in motor vehicle credit: $plug_in_motor_vehicle

Line 6j - Alternative fuel vehicle refueling property credit: $alt_motor_vehicle

Line 7 - Total other nonrefundable credits. Add lines 6d, 6i, and 6j: $TBD

Line 8 - Add lines 1 through 4, 5a, 5b, and 7. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 20: $TBD
"""
