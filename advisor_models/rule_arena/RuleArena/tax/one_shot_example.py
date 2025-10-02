example_gpt = """
Here is an example for you to follow:
<example>
IRS Forms for the tax payer:

# Basic Information

Name: John
Age (on January 2, 2024): 26
Age (on January 2, 2024) of Your Spouse: 28

# Form 1040: U.S. Individual Income Tax Return

Filing Status: qualifying surviving spouse
Using Itemized Deductions: False

Age/Blindness:
* You were born before January 2, 1959: [__]
* You were blind: False
* You spouse was born before January 2, 1959: [__]
* You spouse was blind: False

## Dependents

Qualifying Children: 0

Other Dependents: 0

## Income

Line 1a - Total amount from Form(s) W-2, box 1: $93,845

Line 1b - Household employee wages not reported on Form(s) W-2: $537

Line 1c - Tip income not reported on line 1a: $165

Line 1d - Nontaxable combat pay election: $1,691

Line 1z - Add lines 1a through 1c: [__]

Line 2a - Tax-exempt interest: $1,849

Line 2b - Taxable interest: $3,132

Line 3a - Qualified dividends: $0

Line 3b - Ordinary dividends: $3,406

Line 4a - IRA distributions: $18,662

Line 4b - Taxable amount: $18,236

Line 5a - Pensions and annuities: $33,645

Line 5b - Taxable amount: $3,500

Line 6a - Social security benefits: $7,760

Line 6b - Taxable amount: $5,275

Line 7 - Capital gain or (loss): $0

Line 8 - Additional income from Schedule 1, line 10: [__]

Line 9 - Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income: [__]

Line 10 - Adjustments to income from Schedule 1, line 26: [__]

Line 11 - Subtract line 10 from line 9. This is your adjusted gross income: [__]

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

Line 12 - Standard deduction or itemized deductions (from Schedule A): [__]

Line 13 - Qualified business income deduction from Form 8995 or Form 8995-A: $2,117

Line 14 - Add lines 12 and 13: [__]

Line 15 - Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income: [__]

## Tax and Credits

### Instruction for Line 16

If your taxable income is less than $100,000, you must use the Tax Table, later in these instructions, to figure your tax. Be sure you use the correct column. If your taxable income is $100,000 or more, use the Tax Computation Worksheet right after the Tax Table.

#### 2023 Tax Table

First divide your taxable income by 50, round it down to the nearest integer, then multiply it by 50, and finally add 25. Enter the result (called rounded taxable income) here: [__]

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $11,000 but not over $44,725  | [__]     | × 12% (0.12)     | $ 220.00      | [__]     | [__]     |
| Over $44,725 but not over $95,375  | [__]     | × 22% (0.22)     | $ 4692.50     | [__]     | [__]     |
| Over $95,375 but not over $100,000 | [__]     | × 24% (0.24)     | $ 6600.00     | [__]     | [__]     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $22,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $22,000 but not over $89,450  | [__]     | × 12% (0.12)     | $ 440.00      | [__]     | [__]     |
| Over $89,450 but not over $100,000 | [__]     | × 22% (0.22)     | $ 9385.00     | [__]     | [__]     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $11,000 but not over $44,725  | [__]     | × 12% (0.12)     | $ 220.00      | [__]     | [__]     |
| Over $44,725 but not over $95,375  | [__]     | × 22% (0.22)     | $ 4692.50     | [__]     | [__]     |
| Over $95,375 but not over $100,000 | [__]     | × 24% (0.24)     | $ 6600.00     | [__]     | [__]     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $15,700   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $15,700 but not over $59,850  | [__]     | × 12% (0.12)     | $ 314.00      | [__]     | [__]     |
| Over $59,850 but not over $95,350  | [__]     | × 22% (0.22)     | $ 6299.00     | [__]     | [__]     |
| Over $95,350 but not over $100,000 | [__]     | × 24% (0.24)     | $ 8206.00     | [__]     | [__]     |

#### 2023 Tax Computation Worksheet—Line 16

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 6,600.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 21,168.00     | [__]     | [__]     |
| Over $231,250 but not over $578,125     | [__]     | × 35% (0.35)     | $ 28,105.50     | [__]     | [__]     |
| Over $578,125                           | [__]     | × 37% (0.37)     | $ 39,668.00     | [__]     | [__]     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $190,750 | [__]     | × 22% (0.22)     | $ 9,385.00      | [__]     | [__]     |
| Over $190,750 but not over $364,200     | [__]     | × 24% (0.24)     | $ 13,200.00     | [__]     | [__]     |
| Over $364,200 but not over $462,500     | [__]     | × 32% (0.32)     | $ 42,336.00     | [__]     | [__]     |
| Over $462,500 but not over $693,750     | [__]     | × 35% (0.35)     | $ 56,211.00     | [__]     | [__]     |
| Over $693,750                           | [__]     | × 37% (0.37)     | $ 70086.00      | [__]     | [__]     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 6,600.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 21,168.00     | [__]     | [__]     |
| Over $231,250 but not over $346,875     | [__]     | × 35% (0.35)     | $ 28,105.50     | [__]     | [__]     |
| Over $346,875                           | [__]     | × 37% (0.37)     | $ 35,043.00     | [__]     | [__]     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 8,206.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 22,774.00     | [__]     | [__]     |
| Over $231,250 but not over $578,125     | [__]     | × 35% (0.35)     | $ 29,711.50     | [__]     | [__]     |
| Over $578,125                           | [__]     | × 37% (0.37)     | $ 41,273.50     | [__]     | [__]     |

However, don’t use the Tax Table or Tax Computation Worksheet to figure your tax if the following applies.

Qualified Dividends and Capital Gain Tax Worksheet: Use the Qualified Dividends and Capital Gain Tax Worksheet, later, to figure your tax if you don’t have to use the Schedule D Tax Worksheet and if any of the following applies.
* You reported qualified dividends on Form 1040 or 1040-SR, line 3a.
* You don’t have to file Schedule D and you reported capital gain distributions on Form 1040 or 1040-SR, line 7.
* You are filing Schedule D, and Schedule D, lines 15 and 16, are both more than zero.

#### Qualified Dividends and Capital Gain Tax Worksheet—Line 16

Line 1 - Enter the amount from Form 1040 or 1040-SR, line 15: [__]

Line 2 - Enter the amount from Form 1040 or 1040-SR, line 3a: [__]

Line 3 - Are you filing Schedule D? If Yes, enter the smaller of line 15 or line 16 of Schedule D. If either line 15 or line 16 is blank or a loss, enter -0-. If No, enter the amount from Form 1040 or 1040-SR, line 7: [__]

Line 4 - Add lines 2 and 3: [__]

Line 5 - Subtract line 4 from line 1. If zero or less, enter -0-: [__]

Line 6 - Enter: $44,625 if single or married filing separately, $89,250 if married filing jointly or qualifying surviving spouse, $59,750 if head of household: [__]

Line 7 - Enter the smaller of line 1 or line 6: [__]

Line 8 - Enter the smaller of line 5 or line 7: [__]

Line 9 - Subtract line 8 from line 7. This amount is taxed at 0%: [__]

Line 10 - Enter the smaller of line 1 or line 4: [__]

Line 11 - Enter the amount from line 9: [__]

Line 12 - Subtract line 11 from line 10: [__]

Line 13 - Enter: $492,300 if single, $276,900 if married filing separately, $553,850 if married filing jointly or qualifying surviving spouse, $523,050 if head of household: [__]

Line 14 - Enter the smaller of line 1 or line 13: [__]

Line 15 - Add lines 5 and 9: [__]

Line 16 - Subtract line 15 from line 14. If zero or less, enter -0-: [__]

Line 17 - Enter the smaller of line 12 or line 16: [__]

Line 18 - Multiply line 17 by 15% (0.15): [__]

Line 19 - Add lines 9 and 17: [__]

Line 20 - Subtract line 19 from line 10: [__]

Line 21 - Multiply line 20 by 20% (0.20): [__]

Line 22 - Figure the tax on the amount on line 5. If the amount on line 5 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 5 is $100,000 or more, use the Tax Computation Worksheet: [__]

Line 23 - Add lines 18, 21, and 22: [__]

Line 24 - Figure the tax on the amount on line 1. If the amount on line 1 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 1 is $100,000 or more, use the Tax Computation Worksheet: [__]

Line 25 - Tax on all taxable income. Enter the smaller of line 23 or line 24. Also include this amount on the entry space on Form 1040 or 1040-SR, line 16: [__]

---

Line 16 - Tax (See instructions): [__]

Line 17 - Amount from Schedule 2, line 3: [__]

Line 18 - Add lines 16 and 17: [__]

Line 19 - Child tax credit or credit for other dependents from Schedule 8812: $0

Line 20 - Amount from Schedule 3, line 8: [__]

Line 21 - Add lines 19 and 20: [__]

Line 22 - Subtract line 21 from line 18. If zero or less, enter -0-: [__]

Line 23 - Other taxes, including self-employment tax, from Schedule 2, line 21: [__]

Line 24 - Add lines 22 and 23. This is your total tax: [__]

## Payments

Line 25 - Federal income tax withheld: $17,216

Line 26 - 2023 estimated tax payments and amount applied from 2022 return: $0

Line 27 - Earned income credit (EIC): $0

Line 28 - Additional child tax credit from Schedule 8812: $0

Line 29 - American opportunity credit from Form 8863, line 8: $0

Line 30 - Reserved for future use: $0

Line 31 - Amount from Schedule 3, line 15: [__]

Line 32 - Add lines 27, 28, 29, and 31. These are your total other payments and refundable credits: [__]

Line 33 - Add lines 25, 26, and 32. These are your total payments: [__]

## Amount You Owe

Line 37 - Subtract line 33 from line 24. This is the amount you owe: [__]
    
# Schedule 1 (Form 1040): Additional Income and Adjustments to Income

## Part I: Additional Income

Line 1 - Taxable refunds, credits, or offsets of state and local income taxes: $1,841

Line 2a - Alimony received: $2,953

Line 3 - Business income or (loss). Attach Schedule C: [__]

Line 4 - Other gains or (losses): $35,600

Line 5 - Rental real estate, royalties, partnerships, S corporations, trusts, etc. Attach Schedule E: $0

Line 6 - Farm income or (loss). Attach Schedule F: $0

Line 7 - Unemployment compensation: $4,077

Line 8 - Other income: $1,709

Line 10 - Combine lines 1 through 8. This is your additional income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 8: [__]

## Part II: Adjustments to Income

Line 11 - Educator expenses: $206

Line 12 - Certain business expenses of reservists, performing artists, and fee-basis government officials: $0

Line 13 - Health savings account deduction: $729

Line 14 - Moving expenses for members of the Armed Forces: $0

Line 15 - Deductible part of self-employment tax: $0

Line 16 - Self-employed SEP, SIMPLE, and qualified plans: $0

Line 17 - Self-employed health insurance deduction: $0

Line 18 - Penalty on early withdrawal of savings: $0 

Line 19 - Alimony paid: $0 

Line 20 - IRA deduction: $548

Line 21 - Student loan interest deduction: $1,117

Line 22 - Reserved for future use: $0

Line 23 - Archer MSA deduction: $0

Line 24 - Other adjustments: $870

Line 26 - Add lines 11 through 24. These are your adjustments to income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 10: [__]

# Schedule 2 (Form 1040): Additional Taxes

## Part I: Tax

Line 1 - Alternative minimum tax: $3,272

Line 2 - Excess advance premium tax credit repayment. Attach Form 8962: $744

Line 3 - Add lines 1 and 2. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 17: [__]

## Part II: Other Taxes

Line 4 - Self-employment tax. Attach Schedule SE: [__]

Line 17 - Other additional taxes: $1,445

Line 21 - Add lines 4 and 17. These are your total other taxes. Enter here and on Form 1040 or 1040-SR, line 23, or Form 1040-NR, line 23b: [__]

# Schedule 3 (Form 1040): Additional Credits and Payments

# Part I: Nonrefundable Credits

Line 1 - Foreign tax credit: $304

Line 2 - Credit for child and dependent care expenses from Form 2441, line 11: $684

Line 3 - Education credits from Form 8863, line 19: $0

Line 4 - Retirement savings contributions credit: $594

Line 5a - Residential clean energy credit from Form 5695, line 15: $0

Line 5b - Energy efficient home improvement credit from Form 5695, line 32: $0

Line 6 - Other nonrefundable credits:

Line 6d - Elderly or disabled credit from Schedule R: $117

Line 6i - Qualified plug-in motor vehicle credit: $3,754

Line 6j - Alternative fuel vehicle refueling property credit: $611

Line 7 - Total other nonrefundable credits. Add lines 6d, 6i, and 6j: [__]

Line 8 - Add lines 1 through 4, 5a, 5b, and 7. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 20: [__]

Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:
To calculate the tax owed by John, we will follow the steps outlined in the provided forms and instructions.

### Step 1: Calculate Total Income

1. Line 1z: Add lines 1a through 1c.
   \[
   \text{Line 1z} = \$93,845 + \$537 + \$165 = \$94,547
   \]

2. Line 8: Additional income from Schedule 1, line 10.
   - Schedule 1, Line 10: Combine lines 1 through 8.
     \[
     \text{Line 10} = \$1,841 + \$2,953 + \$35,600 + \$4,077 + \$1,709 = \$46,180
     \]

3. Line 9: Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8.
   \[
   \text{Line 9} = \$94,547 + \$3,132 + \$3,406 + \$18,236 + \$3,500 + \$5,275 + \$0 + \$46,180 = \$174,276
   \]

### Step 2: Calculate Adjusted Gross Income (AGI)

4. Line 10: Adjustments to income from Schedule 1, line 26.
   - Schedule 1, Line 26: Add lines 11 through 24.
     \[
     \text{Line 26} = \$206 + \$729 + \$548 + \$1,117 + \$870 = \$3,470
     \]

5. Line 11: Subtract line 10 from line 9.
   \[
   \text{Line 11} = \$174,276 - \$3,470 = \$170,806
   \]

### Step 3: Calculate Taxable Income

6. Line 12: Standard deduction for Qualifying surviving spouse.
   \[
   \text{Line 12} = \$27,700
   \]

7. Line 13: Qualified business income deduction.
   \[
   \text{Line 13} = \$2,117
   \]

8. Line 14: Add lines 12 and 13.
   \[
   \text{Line 14} = \$27,700 + \$2,117 = \$29,817
   \]

9. Line 15: Subtract line 14 from line 11.
   \[
   \text{Line 15} = \$170,806 - \$29,817 = \$140,989
   \]

### Step 4: Calculate Tax

10. Line 16: Use the Tax Computation Worksheet for Qualifying surviving spouse.
    - Section B: Taxable income is $140,989.
    - Taxable income is over $100,000 but not over $190,750.
    \[
    \text{Tax} = (\$140,989 \times 0.22) - \$9,385 = \$31,017.58 - \$9,385 = \$21,632.58
    \]

### Step 5: Calculate Total Tax

11. Line 17: Amount from Schedule 2, line 3.
    - Schedule 2, Line 3: Add lines 1 and 2.
      \[
      \text{Line 3} = \$3,272 + \$744 = \$4,016
      \]

12. Line 18: Add lines 16 and 17.
    \[
    \text{Line 18} = \$21,632.58 + \$4,016 = \$25,648.58
    \]

13. Line 20: Amount from Schedule 3, line 8.
    - Schedule 3, Line 7: Add lines 6d, 6i, and 6j.
      \[
      \text{Line 7} = \$117 + \$3,754 + \$611 = \$4,482
      \]
    - Schedule 3, Line 8: Add lines 1 through 4, 5a, 5b, and 7.
      \[
      \text{Line 8} = \$304 + \$684 + \$594 + \$4,482 = \$6,064
      \]

14. Line 21: Add lines 19 and 20.
    \[
    \text{Line 21} = \$0 + \$6,064 = \$6,064
    \]

15. Line 22: Subtract line 21 from line 18.
    \[
    \text{Line 22} = \$25,648.58 - \$6,064 = \$19,584.58
    \]

16. Line 23: Other taxes from Schedule 2, line 21.
    - Schedule 2, Line 21: Add lines 4 and 17.
      \[
      \text{Line 21} = \$0 + \$1,445 = \$1,445
      \]

17. Line 24: Add lines 22 and 23.
    \[
    \text{Line 24} = \$19,584.58 + \$1,445 = \$21,029.58
    \]

### Step 6: Calculate Payments and Amount Owed

18. Line 33: Add lines 25, 26, and 32.
    \[
    \text{Line 33} = \$17,216 + \$0 + \$0 = \$17,216
    \]

19. Line 37: Subtract line 33 from line 24.
    \[
    \text{Line 37} = \$21,029.58 - \$17,216 = \$3,813.58
    \]

The total tax owed is $3,813.58.
</example>
"""

example_claude = """
Here is an example for you to follow:
<example>
IRS Forms for the tax payer:
# Basic Information
Name: John
Age (on January 2, 2024): 41
Age (on January 2, 2024) of Your Spouse: 41
# Form 1040: U.S. Individual Income Tax Return
Filing Status: qualifying surviving spouse
Using Itemized Deductions: False
Age/Blindness:
* You were born before January 2, 1959: [__]
* You were blind: True
* You spouse was born before January 2, 1959: [__]
* You spouse was blind: False
## Dependents
Qualifying Children: 0
Other Dependents: 0
## Income
Line 1a - Total amount from Form(s) W-2, box 1: $121,150
Line 1b - Household employee wages not reported on Form(s) W-2: $393
Line 1c - Tip income not reported on line 1a: $4,211
Line 1d - Nontaxable combat pay election: $1,881
Line 1z - Add lines 1a through 1c: [__]
Line 2a - Tax-exempt interest: $2,451
Line 2b - Taxable interest: $3,962
Line 3a - Qualified dividends: $0
Line 3b - Ordinary dividends: $2,494
Line 4a - IRA distributions: $5,912
Line 4b - Taxable amount: $3,663
Line 5a - Pensions and annuities: $13,403
Line 5b - Taxable amount: $7,046
Line 6a - Social security benefits: $28,495
Line 6b - Taxable amount: $4,554
Line 7 - Capital gain or (loss): $0
Line 8 - Additional income from Schedule 1, line 10: [__]
Line 9 - Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income: [__]
Line 10 - Adjustments to income from Schedule 1, line 26: [__]
Line 11 - Subtract line 10 from line 9. This is your adjusted gross income: [__]
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
Line 12 - Standard deduction or itemized deductions (from Schedule A): [__]
Line 13 - Qualified business income deduction from Form 8995 or Form 8995-A: $1,094
Line 14 - Add lines 12 and 13: [__]
Line 15 - Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income: [__]
## Tax and Credits
### Instruction for Line 16
If your taxable income is less than $100,000, you must use the Tax Table, later in these instructions, to figure your tax. Be sure you use the correct column. If your taxable income is $100,000 or more, use the Tax Computation Worksheet right after the Tax Table.
#### 2023 Tax Table
First divide your taxable income by 50, round it down to the nearest integer, then multiply it by 50, and finally add 25. Enter the result (called rounded taxable income) here: [__]
Section A—Use if your filing status is Single. Complete the row below that applies to you.
| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $11,000 but not over $44,725  | [__]     | × 12% (0.12)     | $ 220.00      | [__]     | [__]     |
| Over $44,725 but not over $95,375  | [__]     | × 22% (0.22)     | $ 4692.50     | [__]     | [__]     |
| Over $95,375 but not over $100,000 | [__]     | × 24% (0.24)     | $ 6600.00     | [__]     | [__]     |
Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.
| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $22,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $22,000 but not over $89,450  | [__]     | × 12% (0.12)     | $ 440.00      | [__]     | [__]     |
| Over $89,450 but not over $100,000 | [__]     | × 22% (0.22)     | $ 9385.00     | [__]     | [__]     |
Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.
| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $11,000 but not over $44,725  | [__]     | × 12% (0.12)     | $ 220.00      | [__]     | [__]     |
| Over $44,725 but not over $95,375  | [__]     | × 22% (0.22)     | $ 4692.50     | [__]     | [__]     |
| Over $95,375 but not over $100,000 | [__]     | × 24% (0.24)     | $ 6600.00     | [__]     | [__]     |
Section D—Use if your filing status is Head of household. Complete the row below that applies to you.
| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $15,700   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $15,700 but not over $59,850  | [__]     | × 12% (0.12)     | $ 314.00      | [__]     | [__]     |
| Over $59,850 but not over $95,350  | [__]     | × 22% (0.22)     | $ 6299.00     | [__]     | [__]     |
| Over $95,350 but not over $100,000 | [__]     | × 24% (0.24)     | $ 8206.00     | [__]     | [__]     |
#### 2023 Tax Computation Worksheet—Line 16
Section A—Use if your filing status is Single. Complete the row below that applies to you.
| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 6,600.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 21,168.00     | [__]     | [__]     |
| Over $231,250 but not over $578,125     | [__]     | × 35% (0.35)     | $ 28,105.50     | [__]     | [__]     |
| Over $578,125                           | [__]     | × 37% (0.37)     | $ 39,668.00     | [__]     | [__]     |
Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.
| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $190,750 | [__]     | × 22% (0.22)     | $ 9,385.00      | [__]     | [__]     |
| Over $190,750 but not over $364,200     | [__]     | × 24% (0.24)     | $ 13,200.00     | [__]     | [__]     |
| Over $364,200 but not over $462,500     | [__]     | × 32% (0.32)     | $ 42,336.00     | [__]     | [__]     |
| Over $462,500 but not over $693,750     | [__]     | × 35% (0.35)     | $ 56,211.00     | [__]     | [__]     |
| Over $693,750                           | [__]     | × 37% (0.37)     | $ 70086.00      | [__]     | [__]     |
Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.
| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 6,600.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 21,168.00     | [__]     | [__]     |
| Over $231,250 but not over $346,875     | [__]     | × 35% (0.35)     | $ 28,105.50     | [__]     | [__]     |
| Over $346,875                           | [__]     | × 37% (0.37)     | $ 35,043.00     | [__]     | [__]     |
Section D—Use if your filing status is Head of household. Complete the row below that applies to you.
| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 8,206.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 22,774.00     | [__]     | [__]     |
| Over $231,250 but not over $578,125     | [__]     | × 35% (0.35)     | $ 29,711.50     | [__]     | [__]     |
| Over $578,125                           | [__]     | × 37% (0.37)     | $ 41,273.50     | [__]     | [__]     |
However, don’t use the Tax Table or Tax Computation Worksheet to figure your tax if the following applies.
Qualified Dividends and Capital Gain Tax Worksheet: Use the Qualified Dividends and Capital Gain Tax Worksheet, later, to figure your tax if you don’t have to use the Schedule D Tax Worksheet and if any of the following applies.
* You reported qualified dividends on Form 1040 or 1040-SR, line 3a.
* You don’t have to file Schedule D and you reported capital gain distributions on Form 1040 or 1040-SR, line 7.
* You are filing Schedule D, and Schedule D, lines 15 and 16, are both more than zero.
#### Qualified Dividends and Capital Gain Tax Worksheet—Line 16
Line 1 - Enter the amount from Form 1040 or 1040-SR, line 15: [__]
Line 2 - Enter the amount from Form 1040 or 1040-SR, line 3a: [__]
Line 3 - Are you filing Schedule D? If Yes, enter the smaller of line 15 or line 16 of Schedule D. If either line 15 or line 16 is blank or a loss, enter -0-. If No, enter the amount from Form 1040 or 1040-SR, line 7: [__]
Line 4 - Add lines 2 and 3: [__]
Line 5 - Subtract line 4 from line 1. If zero or less, enter -0-: [__]
Line 6 - Enter: $44,625 if single or married filing separately, $89,250 if married filing jointly or qualifying surviving spouse, $59,750 if head of household: [__]
Line 7 - Enter the smaller of line 1 or line 6: [__]
Line 8 - Enter the smaller of line 5 or line 7: [__]
Line 9 - Subtract line 8 from line 7. This amount is taxed at 0%: [__]
Line 10 - Enter the smaller of line 1 or line 4: [__]
Line 11 - Enter the amount from line 9: [__]
Line 12 - Subtract line 11 from line 10: [__]
Line 13 - Enter: $492,300 if single, $276,900 if married filing separately, $553,850 if married filing jointly or qualifying surviving spouse, $523,050 if head of household: [__]
Line 14 - Enter the smaller of line 1 or line 13: [__]
Line 15 - Add lines 5 and 9: [__]
Line 16 - Subtract line 15 from line 14. If zero or less, enter -0-: [__]
Line 17 - Enter the smaller of line 12 or line 16: [__]
Line 18 - Multiply line 17 by 15% (0.15): [__]
Line 19 - Add lines 9 and 17: [__]
Line 20 - Subtract line 19 from line 10: [__]
Line 21 - Multiply line 20 by 20% (0.20): [__]
Line 22 - Figure the tax on the amount on line 5. If the amount on line 5 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 5 is $100,000 or more, use the Tax Computation Worksheet: [__]
Line 23 - Add lines 18, 21, and 22: [__]
Line 24 - Figure the tax on the amount on line 1. If the amount on line 1 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 1 is $100,000 or more, use the Tax Computation Worksheet: [__]
Line 25 - Tax on all taxable income. Enter the smaller of line 23 or line 24. Also include this amount on the entry space on Form 1040 or 1040-SR, line 16: [__]
---
Line 16 - Tax (See instructions): [__]
Line 17 - Amount from Schedule 2, line 3: [__]
Line 18 - Add lines 16 and 17: [__]
Line 19 - Child tax credit or credit for other dependents from Schedule 8812: $0
Line 20 - Amount from Schedule 3, line 8: [__]
Line 21 - Add lines 19 and 20: [__]
Line 22 - Subtract line 21 from line 18. If zero or less, enter -0-: [__]
Line 23 - Other taxes, including self-employment tax, from Schedule 2, line 21: [__]
Line 24 - Add lines 22 and 23. This is your total tax: [__]
## Payments
Line 25 - Federal income tax withheld: $18,930
Line 26 - 2023 estimated tax payments and amount applied from 2022 return: $0
Line 27 - Earned income credit (EIC): $0
Line 28 - Additional child tax credit from Schedule 8812: $0
Line 29 - American opportunity credit from Form 8863, line 8: $0
Line 30 - Reserved for future use: $0
Line 31 - Amount from Schedule 3, line 15: [__]
Line 32 - Add lines 27, 28, 29, and 31. These are your total other payments and refundable credits: [__]
Line 33 - Add lines 25, 26, and 32. These are your total payments: [__]
## Amount You Owe
Line 37 - Subtract line 33 from line 24. This is the amount you owe: [__]
    
# Schedule 1 (Form 1040): Additional Income and Adjustments to Income
## Part I: Additional Income
Line 1 - Taxable refunds, credits, or offsets of state and local income taxes: $987
Line 2a - Alimony received: $2,311
Line 3 - Business income or (loss). Attach Schedule C: [__]
Line 4 - Other gains or (losses): $43,253
Line 5 - Rental real estate, royalties, partnerships, S corporations, trusts, etc. Attach Schedule E: $0
Line 6 - Farm income or (loss). Attach Schedule F: $0
Line 7 - Unemployment compensation: $3,920
Line 8 - Other income: $721
Line 10 - Combine lines 1 through 8. This is your additional income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 8: [__]
## Part II: Adjustments to Income
Line 11 - Educator expenses: $168
Line 12 - Certain business expenses of reservists, performing artists, and fee-basis government officials: $0
Line 13 - Health savings account deduction: $2,935
Line 14 - Moving expenses for members of the Armed Forces: $0
Line 15 - Deductible part of self-employment tax: $0
Line 16 - Self-employed SEP, SIMPLE, and qualified plans: $0
Line 17 - Self-employed health insurance deduction: $0
Line 18 - Penalty on early withdrawal of savings: $0 
Line 19 - Alimony paid: $0 
Line 20 - IRA deduction: $3,261
Line 21 - Student loan interest deduction: $1,233
Line 22 - Reserved for future use: $0
Line 23 - Archer MSA deduction: $0
Line 24 - Other adjustments: $630
Line 26 - Add lines 11 through 24. These are your adjustments to income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 10: [__]
# Schedule 2 (Form 1040): Additional Taxes
## Part I: Tax
Line 1 - Alternative minimum tax: $4,500
Line 2 - Excess advance premium tax credit repayment. Attach Form 8962: $617
Line 3 - Add lines 1 and 2. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 17: [__]
## Part II: Other Taxes
Line 4 - Self-employment tax. Attach Schedule SE: [__]
Line 17 - Other additional taxes: $611
Line 21 - Add lines 4 and 17. These are your total other taxes. Enter here and on Form 1040 or 1040-SR, line 23, or Form 1040-NR, line 23b: [__]
# Schedule 3 (Form 1040): Additional Credits and Payments
# Part I: Nonrefundable Credits
Line 1 - Foreign tax credit: $672
Line 2 - Credit for child and dependent care expenses from Form 2441, line 11: $1,856
Line 3 - Education credits from Form 8863, line 19: $0
Line 4 - Retirement savings contributions credit: $192
Line 5a - Residential clean energy credit from Form 5695, line 15: $0
Line 5b - Energy efficient home improvement credit from Form 5695, line 32: $0
Line 6 - Other nonrefundable credits:
Line 6d - Elderly or disabled credit from Schedule R: $278
Line 6i - Qualified plug-in motor vehicle credit: $1,811
Line 6j - Alternative fuel vehicle refueling property credit: $913
Line 7 - Total other nonrefundable credits. Add lines 6d, 6i, and 6j: [__]
Line 8 - Add lines 1 through 4, 5a, 5b, and 7. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 20: [__]
Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:
Let me help calculate the tax step by step.
Form 1040:
1. Line 1z = Line 1a + Line 1b + Line 1c = $121,150 + $393 + $4,211 = $125,754
2. Line 8 (from Schedule 1, Line 10) = $987 + $2,311 + $0 + $43,253 + $0 + $0 + $3,920 + $721 = $51,192 (sum of Schedule 1 Lines 1-8)
3. Line 9 (total income) = Line 1z + Line 2b + Line 3b + Line 4b + Line 5b + Line 6b + Line 7 + Line 8 = $125,754 + $3,962 + $2,494 + $3,663 + $7,046 + $4,554 + $0 + $51,192 = $198,665
4. Line 10 (from Schedule 1, line 26) = $168 + $0 + $2,935 + $0 + $0 + $0 + $0 + $0 + $0 + $3,261 + $1,233 + $0 + $0 + $630 = $8,227 (sum of Schedule 1 Lines 11-24)
5. Line 11 (AGI) = Line 9 - Line 10 = $198,665 - $8,227 = $190,438
For Line 12 (Standard Deduction):
- Filing status is qualifying surviving spouse
- Taxpayer is blind (1 box checked)
- Using Standard Deduction Chart: $29,200
6. Line 14 = Line 12 + Line 13 = $29,200 + $1,094 = $30,294
7. Line 15 (taxable income) = Line 11 - Line 14 = $190,438 - $30,294 = $160,144
For Line 16 (Tax):
Since taxable income is over $100,000, using Tax Computation Worksheet:
- Filing status is qualifying surviving spouse, using Tax Computation Worksheet Section B
- $160,144 falls in "At least $100,000 but not over $190,750" bracket
- Tax = ($160,144 × 0.22) - $9,385 = $25,846.68
8. Line 17 (from Schedule 2, Line 3) = $4,500 + $617 = $5,117 (sum of Schedule 2 Lines 1 and 2)
9. Line 18 = Line 16 + Line 17 = $25,846.68 + $5,117 = $30,963.68
10. Line 20 (from Schedule 3, Line 8):
- Schedule 3, Line 7 = (Schedule 3) Line 6d + Line 6i + Line 6j = $278 + $1,811 + $913 = $3,002
- Schedule 3, Line 8 = (Schedule 3) Line 1 + Line 2 + Line 3 + Line 4 + Line 5a + Line 5b + Line 7 = $672 + $1,856 + $0 + $192 + $0 + $0 + $3,002 = $5,722 (sum of Schedule 3 Lines 1-7)
11. Line 21 = Line 19 + Line 20 = $0 + $5,722 = $5,722
12. Line 22 = Line 18 - Line 21 = $30,963.68 - $5,722 = $25,241.68
13. Line 23 (from Schedule 2, Line 21) = $0 + $611 = $611 (sum of Schedule 2 Lines 4 and 17)
14. Line 24 (total tax) = Line 22 + Line 23 = $25,241.68 + $611 = $25,852.68
15. Line 32 = Line 27 + Line 28 + Line 29 + Line 31 = $0 + $0 + $0 + $0 = $0
16. Line 33 (total payments) = Line 25 + Line 26 + Line 32 = $18,930 + $0 + $0 = $18,930
17. Line 37 (amount owed) = Line 24 - Line 33 = $25,852.68 - $18,930 = $6,922.68
The total tax owed is $6,922.68.
</example>
"""

example_llama = """
Here is an example for you to follow:
<example>
IRS Forms for the tax payer:

# Basic Information

Name: John
Age (on January 2, 2024): 38
Age (on January 2, 2024) of Your Spouse: 38

# Form 1040: U.S. Individual Income Tax Return

Filing Status: married filing separately
Using Itemized Deductions: False

Age/Blindness:
* You were born before January 2, 1959: [__]
* You were blind: False
* You spouse was born before January 2, 1959: [__]
* You spouse was blind: False

## Dependents

Qualifying Children: 0

Other Dependents: 0

## Income

Line 1a - Total amount from Form(s) W-2, box 1: $35,162

Line 1b - Household employee wages not reported on Form(s) W-2: $1,909

Line 1c - Tip income not reported on line 1a: $138

Line 1d - Nontaxable combat pay election: $8,632

Line 1z - Add lines 1a through 1c: [__]

Line 2a - Tax-exempt interest: $4,076

Line 2b - Taxable interest: $2,735

Line 3a - Qualified dividends: $0

Line 3b - Ordinary dividends: $4,024

Line 4a - IRA distributions: $6

Line 4b - Taxable amount: $4

Line 5a - Pensions and annuities: $10,903

Line 5b - Taxable amount: $2,235

Line 6a - Social security benefits: $28,869

Line 6b - Taxable amount: $10,138

Line 7 - Capital gain or (loss): $0

Line 8 - Additional income from Schedule 1, line 10: [__]

Line 9 - Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income: [__]

Line 10 - Adjustments to income from Schedule 1, line 26: [__]

Line 11 - Subtract line 10 from line 9. This is your adjusted gross income: [__]

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

Line 12 - Standard deduction or itemized deductions (from Schedule A): [__]

Line 13 - Qualified business income deduction from Form 8995 or Form 8995-A: $7,964

Line 14 - Add lines 12 and 13: [__]

Line 15 - Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income: [__]

## Tax and Credits

### Instruction for Line 16

If your taxable income is less than $100,000, you must use the Tax Table, later in these instructions, to figure your tax. Be sure you use the correct column. If your taxable income is $100,000 or more, use the Tax Computation Worksheet right after the Tax Table.

#### 2023 Tax Table

First divide your taxable income by 50, round it down to the nearest integer, then multiply it by 50, and finally add 25. Enter the result (called rounded taxable income) here: [__]

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $11,000 but not over $44,725  | [__]     | × 12% (0.12)     | $ 220.00      | [__]     | [__]     |
| Over $44,725 but not over $95,375  | [__]     | × 22% (0.22)     | $ 4692.50     | [__]     | [__]     |
| Over $95,375 but not over $100,000 | [__]     | × 24% (0.24)     | $ 6600.00     | [__]     | [__]     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $22,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $22,000 but not over $89,450  | [__]     | × 12% (0.12)     | $ 440.00      | [__]     | [__]     |
| Over $89,450 but not over $100,000 | [__]     | × 22% (0.22)     | $ 9385.00     | [__]     | [__]     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $11,000 but not over $44,725  | [__]     | × 12% (0.12)     | $ 220.00      | [__]     | [__]     |
| Over $44,725 but not over $95,375  | [__]     | × 22% (0.22)     | $ 4692.50     | [__]     | [__]     |
| Over $95,375 but not over $100,000 | [__]     | × 24% (0.24)     | $ 6600.00     | [__]     | [__]     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $15,700   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $15,700 but not over $59,850  | [__]     | × 12% (0.12)     | $ 314.00      | [__]     | [__]     |
| Over $59,850 but not over $95,350  | [__]     | × 22% (0.22)     | $ 6299.00     | [__]     | [__]     |
| Over $95,350 but not over $100,000 | [__]     | × 24% (0.24)     | $ 8206.00     | [__]     | [__]     |

#### 2023 Tax Computation Worksheet—Line 16

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 6,600.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 21,168.00     | [__]     | [__]     |
| Over $231,250 but not over $578,125     | [__]     | × 35% (0.35)     | $ 28,105.50     | [__]     | [__]     |
| Over $578,125                           | [__]     | × 37% (0.37)     | $ 39,668.00     | [__]     | [__]     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $190,750 | [__]     | × 22% (0.22)     | $ 9,385.00      | [__]     | [__]     |
| Over $190,750 but not over $364,200     | [__]     | × 24% (0.24)     | $ 13,200.00     | [__]     | [__]     |
| Over $364,200 but not over $462,500     | [__]     | × 32% (0.32)     | $ 42,336.00     | [__]     | [__]     |
| Over $462,500 but not over $693,750     | [__]     | × 35% (0.35)     | $ 56,211.00     | [__]     | [__]     |
| Over $693,750                           | [__]     | × 37% (0.37)     | $ 70086.00      | [__]     | [__]     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 6,600.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 21,168.00     | [__]     | [__]     |
| Over $231,250 but not over $346,875     | [__]     | × 35% (0.35)     | $ 28,105.50     | [__]     | [__]     |
| Over $346,875                           | [__]     | × 37% (0.37)     | $ 35,043.00     | [__]     | [__]     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 8,206.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 22,774.00     | [__]     | [__]     |
| Over $231,250 but not over $578,125     | [__]     | × 35% (0.35)     | $ 29,711.50     | [__]     | [__]     |
| Over $578,125                           | [__]     | × 37% (0.37)     | $ 41,273.50     | [__]     | [__]     |

However, don’t use the Tax Table or Tax Computation Worksheet to figure your tax if the following applies.

Qualified Dividends and Capital Gain Tax Worksheet: Use the Qualified Dividends and Capital Gain Tax Worksheet, later, to figure your tax if you don’t have to use the Schedule D Tax Worksheet and if any of the following applies.
* You reported qualified dividends on Form 1040 or 1040-SR, line 3a.
* You don’t have to file Schedule D and you reported capital gain distributions on Form 1040 or 1040-SR, line 7.
* You are filing Schedule D, and Schedule D, lines 15 and 16, are both more than zero.

#### Qualified Dividends and Capital Gain Tax Worksheet—Line 16

Line 1 - Enter the amount from Form 1040 or 1040-SR, line 15: [__]

Line 2 - Enter the amount from Form 1040 or 1040-SR, line 3a: [__]

Line 3 - Are you filing Schedule D? If Yes, enter the smaller of line 15 or line 16 of Schedule D. If either line 15 or line 16 is blank or a loss, enter -0-. If No, enter the amount from Form 1040 or 1040-SR, line 7: [__]

Line 4 - Add lines 2 and 3: [__]

Line 5 - Subtract line 4 from line 1. If zero or less, enter -0-: [__]

Line 6 - Enter: $44,625 if single or married filing separately, $89,250 if married filing jointly or qualifying surviving spouse, $59,750 if head of household: [__]

Line 7 - Enter the smaller of line 1 or line 6: [__]

Line 8 - Enter the smaller of line 5 or line 7: [__]

Line 9 - Subtract line 8 from line 7. This amount is taxed at 0%: [__]

Line 10 - Enter the smaller of line 1 or line 4: [__]

Line 11 - Enter the amount from line 9: [__]

Line 12 - Subtract line 11 from line 10: [__]

Line 13 - Enter: $492,300 if single, $276,900 if married filing separately, $553,850 if married filing jointly or qualifying surviving spouse, $523,050 if head of household: [__]

Line 14 - Enter the smaller of line 1 or line 13: [__]

Line 15 - Add lines 5 and 9: [__]

Line 16 - Subtract line 15 from line 14. If zero or less, enter -0-: [__]

Line 17 - Enter the smaller of line 12 or line 16: [__]

Line 18 - Multiply line 17 by 15% (0.15): [__]

Line 19 - Add lines 9 and 17: [__]

Line 20 - Subtract line 19 from line 10: [__]

Line 21 - Multiply line 20 by 20% (0.20): [__]

Line 22 - Figure the tax on the amount on line 5. If the amount on line 5 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 5 is $100,000 or more, use the Tax Computation Worksheet: [__]

Line 23 - Add lines 18, 21, and 22: [__]

Line 24 - Figure the tax on the amount on line 1. If the amount on line 1 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 1 is $100,000 or more, use the Tax Computation Worksheet: [__]

Line 25 - Tax on all taxable income. Enter the smaller of line 23 or line 24. Also include this amount on the entry space on Form 1040 or 1040-SR, line 16: [__]

---

Line 16 - Tax (See instructions): [__]

Line 17 - Amount from Schedule 2, line 3: [__]

Line 18 - Add lines 16 and 17: [__]

Line 19 - Child tax credit or credit for other dependents from Schedule 8812: $0

Line 20 - Amount from Schedule 3, line 8: [__]

Line 21 - Add lines 19 and 20: [__]

Line 22 - Subtract line 21 from line 18. If zero or less, enter -0-: [__]

Line 23 - Other taxes, including self-employment tax, from Schedule 2, line 21: [__]

Line 24 - Add lines 22 and 23. This is your total tax: [__]

## Payments

Line 25 - Federal income tax withheld: $3,652

Line 26 - 2023 estimated tax payments and amount applied from 2022 return: $0

Line 27 - Earned income credit (EIC): $0

Line 28 - Additional child tax credit from Schedule 8812: $0

Line 29 - American opportunity credit from Form 8863, line 8: $0

Line 30 - Reserved for future use: $0

Line 31 - Amount from Schedule 3, line 15: [__]

Line 32 - Add lines 27, 28, 29, and 31. These are your total other payments and refundable credits: [__]

Line 33 - Add lines 25, 26, and 32. These are your total payments: [__]

## Amount You Owe

Line 37 - Subtract line 33 from line 24. This is the amount you owe: [__]
    
# Schedule 1 (Form 1040): Additional Income and Adjustments to Income

## Part I: Additional Income

Line 1 - Taxable refunds, credits, or offsets of state and local income taxes: $1,488

Line 2a - Alimony received: $4,549

Line 3 - Business income or (loss). Attach Schedule C: [__]

Line 4 - Other gains or (losses): $46,064

Line 5 - Rental real estate, royalties, partnerships, S corporations, trusts, etc. Attach Schedule E: $0

Line 6 - Farm income or (loss). Attach Schedule F: $0

Line 7 - Unemployment compensation: $9,157

Line 8 - Other income: $888

Line 10 - Combine lines 1 through 8. This is your additional income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 8: [__]

## Part II: Adjustments to Income

Line 11 - Educator expenses: $60

Line 12 - Certain business expenses of reservists, performing artists, and fee-basis government officials: $0

Line 13 - Health savings account deduction: $1,229

Line 14 - Moving expenses for members of the Armed Forces: $0

Line 15 - Deductible part of self-employment tax: $0

Line 16 - Self-employed SEP, SIMPLE, and qualified plans: $0

Line 17 - Self-employed health insurance deduction: $0

Line 18 - Penalty on early withdrawal of savings: $0 

Line 19 - Alimony paid: $0 

Line 20 - IRA deduction: $843

Line 21 - Student loan interest deduction: $591

Line 22 - Reserved for future use: $0

Line 23 - Archer MSA deduction: $0

Line 24 - Other adjustments: $167

Line 26 - Add lines 11 through 24. These are your adjustments to income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 10: [__]

# Schedule 2 (Form 1040): Additional Taxes

## Part I: Tax

Line 1 - Alternative minimum tax: $2,508

Line 2 - Excess advance premium tax credit repayment. Attach Form 8962: $198

Line 3 - Add lines 1 and 2. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 17: [__]

## Part II: Other Taxes

Line 4 - Self-employment tax. Attach Schedule SE: [__]

Line 17 - Other additional taxes: $67

Line 21 - Add lines 4 and 17. These are your total other taxes. Enter here and on Form 1040 or 1040-SR, line 23, or Form 1040-NR, line 23b: [__]

# Schedule 3 (Form 1040): Additional Credits and Payments

# Part I: Nonrefundable Credits

Line 1 - Foreign tax credit: $659

Line 2 - Credit for child and dependent care expenses from Form 2441, line 11: $1,113

Line 3 - Education credits from Form 8863, line 19: $0

Line 4 - Retirement savings contributions credit: $900

Line 5a - Residential clean energy credit from Form 5695, line 15: $0

Line 5b - Energy efficient home improvement credit from Form 5695, line 32: $0

Line 6 - Other nonrefundable credits:

Line 6d - Elderly or disabled credit from Schedule R: $419

Line 6i - Qualified plug-in motor vehicle credit: $2,938

Line 6j - Alternative fuel vehicle refueling property credit: $1,891

Line 7 - Total other nonrefundable credits. Add lines 6d, 6i, and 6j: [__]

Line 8 - Add lines 1 through 4, 5a, 5b, and 7. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 20: [__]

Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:
To calculate the tax owed by the payer, we will follow the instructions provided by the forms.

Form 1040

## Income

Line 1z - Add lines 1a through 1c: $35,162 + $1,909 + $138 = $37,209

Line 9 - Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. 
First, we need to calculate Line 8 - Additional income from Schedule 1, line 10.

Schedule 1 (Form 1040)

## Part I: Additional Income

Line 10 - Combine lines 1 through 8: $1,488 + $4,549 + $0 + $46,064 + $0 + $0 + $9,157 + $888 = $62,146

Now we can calculate Line 9 of Form 1040:
$37,209 + $2,735 + $4,024 + $4 + $2,235 + $10,138 + $0 + $62,146 = $118,491

Line 10 - Adjustments to income from Schedule 1, line 26.

Schedule 1 (Form 1040)

## Part II: Adjustments to Income

Line 26 - Add lines 11 through 24: $60 + $0 + $1,229 + $0 + $0 + $0 + $0 + $0 + $0 + $843 + $591 + $0 + $0 + $167 = $2,890

Now we can calculate Line 11 of Form 1040:
$118,491 - $2,890 = $115,601

Line 12 - Standard deduction or itemized deductions (from Schedule A): 
Since the payer is married filing separately, the standard deduction is $13,850.

Line 14 - Add lines 12 and 13: $13,850 + $7,964 = $21,814

Line 15 - Subtract line 14 from line 11: $115,601 - $21,814 = $93,787

## Tax and Credits

Since the payer's taxable income is less than $100,000, we use the Tax Table to figure the tax.

2023 Tax Table

Section C—Use if your filing status is Married filing separately.

First, we divide the taxable income by 50, round it down to the nearest integer, then multiply it by 50, and finally add 25.
$93,787 / 50 = 1875.74, rounded down to 1875, then multiplied by 50 and added 25: 1875 * 50 + 25 = $93,775

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| Over $44,725 but not over $95,375  | $93,775  | × 22% (0.22)     | $20,630.50    | $4,692.50 | $15,938.00 |

Line 16 - Tax: $15,938.00

Line 17 - Amount from Schedule 2, line 3: 

Schedule 2 (Form 1040)

## Part I: Tax

Line 3 - Add lines 1 and 2: $2,508 + $198 = $2,706

Now we can calculate Line 18 of Form 1040:
$15,938 + $2,706 = $18,644

Line 20 - Amount from Schedule 3, line 8: 

Schedule 3 (Form 1040)

## Part I: Nonrefundable Credits

Line 7 - Total other nonrefundable credits. Add lines 6d, 6i, and 6j: $419 + $2,938 + $1,891 = $5,248

Line 8 - Add lines 1 through 4, 5a, 5b, and 7: $659 + $1,113 + $0 + $900 + $0 + $0 + $5,248 = $7,920

Now we can calculate Line 21 of Form 1040:
$0 + $7,920 = $7,920

Line 22 - Subtract line 21 from line 18: $18,644 - $7,920 = $10,724

Line 23 - Other taxes, including self-employment tax, from Schedule 2, line 21: 

Schedule 2 (Form 1040)

## Part II: Other Taxes

Line 21 - Add lines 4 and 17: $0 + $67 = $67

Now we can calculate Line 24 of Form 1040:
$10,724 + $67 = $10,791

## Payments

Line 33 - Add lines 25, 26, and 32: $3,652 + $0 + $0 = $3,652

## Amount You Owe

Line 37 - Subtract line 33 from line 24: $10,791 - $3,652 = $7,139

The total tax owed is $7,139.
</example>
"""

example_qwen = """
Here is an example for you to follow:
<example>
IRS Forms for the tax payer:

# Basic Information

Name: John
Age (on January 2, 2024): 68
Age (on January 2, 2024) of Your Spouse: 68

# Form 1040: U.S. Individual Income Tax Return

Filing Status: head of household
Using Itemized Deductions: False

Age/Blindness:
* You were born before January 2, 1959: [__]
* You were blind: False
* You spouse was born before January 2, 1959: [__]
* You spouse was blind: False

## Dependents

Qualifying Children: 0

Other Dependents: 0

## Income

Line 1a - Total amount from Form(s) W-2, box 1: $129,500

Line 1b - Household employee wages not reported on Form(s) W-2: $923

Line 1c - Tip income not reported on line 1a: $2,144

Line 1d - Nontaxable combat pay election: $210

Line 1z - Add lines 1a through 1c: [__]

Line 2a - Tax-exempt interest: $1,318

Line 2b - Taxable interest: $1,343

Line 3a - Qualified dividends: $0

Line 3b - Ordinary dividends: $1,650

Line 4a - IRA distributions: $5,531

Line 4b - Taxable amount: $1,023

Line 5a - Pensions and annuities: $27,868

Line 5b - Taxable amount: $2,679

Line 6a - Social security benefits: $9,403

Line 6b - Taxable amount: $1,176

Line 7 - Capital gain or (loss): $0

Line 8 - Additional income from Schedule 1, line 10: [__]

Line 9 - Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income: [__]

Line 10 - Adjustments to income from Schedule 1, line 26: [__]

Line 11 - Subtract line 10 from line 9. This is your adjusted gross income: [__]

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

Line 12 - Standard deduction or itemized deductions (from Schedule A): [__]

Line 13 - Qualified business income deduction from Form 8995 or Form 8995-A: $6,727

Line 14 - Add lines 12 and 13: [__]

Line 15 - Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income: [__]

## Tax and Credits

### Instruction for Line 16

If your taxable income is less than $100,000, you must use the Tax Table, later in these instructions, to figure your tax. Be sure you use the correct column. If your taxable income is $100,000 or more, use the Tax Computation Worksheet right after the Tax Table.

#### 2023 Tax Table

First divide your taxable income by 50, round it down to the nearest integer, then multiply it by 50, and finally add 25. Enter the result (called rounded taxable income) here: [__]

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $11,000 but not over $44,725  | [__]     | × 12% (0.12)     | $ 220.00      | [__]     | [__]     |
| Over $44,725 but not over $95,375  | [__]     | × 22% (0.22)     | $ 4692.50     | [__]     | [__]     |
| Over $95,375 but not over $100,000 | [__]     | × 24% (0.24)     | $ 6600.00     | [__]     | [__]     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $22,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $22,000 but not over $89,450  | [__]     | × 12% (0.12)     | $ 440.00      | [__]     | [__]     |
| Over $89,450 but not over $100,000 | [__]     | × 22% (0.22)     | $ 9385.00     | [__]     | [__]     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $11,000   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $11,000 but not over $44,725  | [__]     | × 12% (0.12)     | $ 220.00      | [__]     | [__]     |
| Over $44,725 but not over $95,375  | [__]     | × 22% (0.22)     | $ 4692.50     | [__]     | [__]     |
| Over $95,375 but not over $100,000 | [__]     | × 24% (0.24)     | $ 6600.00     | [__]     | [__]     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| If your rounded taxable income is— | (a) Enter the amount of rounded taxable income | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| ---------------------------------- | -------- | ---------------- | ------------- | -------- | -------- |
| At least $0 but not over $15,700   | [__]     | × 10% (0.10)     | $ 0           | [__]     | [__]     |
| Over $15,700 but not over $59,850  | [__]     | × 12% (0.12)     | $ 314.00      | [__]     | [__]     |
| Over $59,850 but not over $95,350  | [__]     | × 22% (0.22)     | $ 6299.00     | [__]     | [__]     |
| Over $95,350 but not over $100,000 | [__]     | × 24% (0.24)     | $ 8206.00     | [__]     | [__]     |

#### 2023 Tax Computation Worksheet—Line 16

Section A—Use if your filing status is Single. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 6,600.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 21,168.00     | [__]     | [__]     |
| Over $231,250 but not over $578,125     | [__]     | × 35% (0.35)     | $ 28,105.50     | [__]     | [__]     |
| Over $578,125                           | [__]     | × 37% (0.37)     | $ 39,668.00     | [__]     | [__]     |

Section B—Use if your filing status is Married filing jointly or Qualifying surviving spouse. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $190,750 | [__]     | × 22% (0.22)     | $ 9,385.00      | [__]     | [__]     |
| Over $190,750 but not over $364,200     | [__]     | × 24% (0.24)     | $ 13,200.00     | [__]     | [__]     |
| Over $364,200 but not over $462,500     | [__]     | × 32% (0.32)     | $ 42,336.00     | [__]     | [__]     |
| Over $462,500 but not over $693,750     | [__]     | × 35% (0.35)     | $ 56,211.00     | [__]     | [__]     |
| Over $693,750                           | [__]     | × 37% (0.37)     | $ 70086.00      | [__]     | [__]     |

Section C—Use if your filing status is Married filing separately. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 6,600.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 21,168.00     | [__]     | [__]     |
| Over $231,250 but not over $346,875     | [__]     | × 35% (0.35)     | $ 28,105.50     | [__]     | [__]     |
| Over $346,875                           | [__]     | × 37% (0.37)     | $ 35,043.00     | [__]     | [__]     |

Section D—Use if your filing status is Head of household. Complete the row below that applies to you.

| Taxable income. If line 15 is— | (a) Enter the amount from line 15 | (b) Multiplication amount | (c) Multiply  (a) by (b) | (d) Subtraction amount | Tax. Subtract (d) from (c). Enter the result here and on the entry space on line 16. |
| --------------------------------------- | -------- | ---------------- | --------------- | -------- | -------- |
| At least $100,000 but not over $182,100 | [__]     | × 24% (0.24)     | $ 8,206.00      | [__]     | [__]     |
| Over $182,100 but not over $231,250     | [__]     | × 32% (0.32)     | $ 22,774.00     | [__]     | [__]     |
| Over $231,250 but not over $578,125     | [__]     | × 35% (0.35)     | $ 29,711.50     | [__]     | [__]     |
| Over $578,125                           | [__]     | × 37% (0.37)     | $ 41,273.50     | [__]     | [__]     |

However, don’t use the Tax Table or Tax Computation Worksheet to figure your tax if the following applies.

Qualified Dividends and Capital Gain Tax Worksheet: Use the Qualified Dividends and Capital Gain Tax Worksheet, later, to figure your tax if you don’t have to use the Schedule D Tax Worksheet and if any of the following applies.
* You reported qualified dividends on Form 1040 or 1040-SR, line 3a.
* You don’t have to file Schedule D and you reported capital gain distributions on Form 1040 or 1040-SR, line 7.
* You are filing Schedule D, and Schedule D, lines 15 and 16, are both more than zero.

#### Qualified Dividends and Capital Gain Tax Worksheet—Line 16

Line 1 - Enter the amount from Form 1040 or 1040-SR, line 15: [__]

Line 2 - Enter the amount from Form 1040 or 1040-SR, line 3a: [__]

Line 3 - Are you filing Schedule D? If Yes, enter the smaller of line 15 or line 16 of Schedule D. If either line 15 or line 16 is blank or a loss, enter -0-. If No, enter the amount from Form 1040 or 1040-SR, line 7: [__]

Line 4 - Add lines 2 and 3: [__]

Line 5 - Subtract line 4 from line 1. If zero or less, enter -0-: [__]

Line 6 - Enter: $44,625 if single or married filing separately, $89,250 if married filing jointly or qualifying surviving spouse, $59,750 if head of household: [__]

Line 7 - Enter the smaller of line 1 or line 6: [__]

Line 8 - Enter the smaller of line 5 or line 7: [__]

Line 9 - Subtract line 8 from line 7. This amount is taxed at 0%: [__]

Line 10 - Enter the smaller of line 1 or line 4: [__]

Line 11 - Enter the amount from line 9: [__]

Line 12 - Subtract line 11 from line 10: [__]

Line 13 - Enter: $492,300 if single, $276,900 if married filing separately, $553,850 if married filing jointly or qualifying surviving spouse, $523,050 if head of household: [__]

Line 14 - Enter the smaller of line 1 or line 13: [__]

Line 15 - Add lines 5 and 9: [__]

Line 16 - Subtract line 15 from line 14. If zero or less, enter -0-: [__]

Line 17 - Enter the smaller of line 12 or line 16: [__]

Line 18 - Multiply line 17 by 15% (0.15): [__]

Line 19 - Add lines 9 and 17: [__]

Line 20 - Subtract line 19 from line 10: [__]

Line 21 - Multiply line 20 by 20% (0.20): [__]

Line 22 - Figure the tax on the amount on line 5. If the amount on line 5 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 5 is $100,000 or more, use the Tax Computation Worksheet: [__]

Line 23 - Add lines 18, 21, and 22: [__]

Line 24 - Figure the tax on the amount on line 1. If the amount on line 1 is less than $100,000, use the Tax Table to figure the tax. If the amount on line 1 is $100,000 or more, use the Tax Computation Worksheet: [__]

Line 25 - Tax on all taxable income. Enter the smaller of line 23 or line 24. Also include this amount on the entry space on Form 1040 or 1040-SR, line 16: [__]

---

Line 16 - Tax (See instructions): [__]

Line 17 - Amount from Schedule 2, line 3: [__]

Line 18 - Add lines 16 and 17: [__]

Line 19 - Child tax credit or credit for other dependents from Schedule 8812: $0

Line 20 - Amount from Schedule 3, line 8: [__]

Line 21 - Add lines 19 and 20: [__]

Line 22 - Subtract line 21 from line 18. If zero or less, enter -0-: [__]

Line 23 - Other taxes, including self-employment tax, from Schedule 2, line 21: [__]

Line 24 - Add lines 22 and 23. This is your total tax: [__]

## Payments

Line 25 - Federal income tax withheld: $5,512

Line 26 - 2023 estimated tax payments and amount applied from 2022 return: $0

Line 27 - Earned income credit (EIC): $0

Line 28 - Additional child tax credit from Schedule 8812: $0

Line 29 - American opportunity credit from Form 8863, line 8: $0

Line 30 - Reserved for future use: $0

Line 31 - Amount from Schedule 3, line 15: [__]

Line 32 - Add lines 27, 28, 29, and 31. These are your total other payments and refundable credits: [__]

Line 33 - Add lines 25, 26, and 32. These are your total payments: [__]

## Amount You Owe

Line 37 - Subtract line 33 from line 24. This is the amount you owe: [__]
    
# Schedule 1 (Form 1040): Additional Income and Adjustments to Income

## Part I: Additional Income

Line 1 - Taxable refunds, credits, or offsets of state and local income taxes: $1,135

Line 2a - Alimony received: $2,864

Line 3 - Business income or (loss). Attach Schedule C: [__]

Line 4 - Other gains or (losses): $21,053

Line 5 - Rental real estate, royalties, partnerships, S corporations, trusts, etc. Attach Schedule E: $0

Line 6 - Farm income or (loss). Attach Schedule F: $0

Line 7 - Unemployment compensation: $1,280

Line 8 - Other income: $337

Line 10 - Combine lines 1 through 8. This is your additional income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 8: [__]

## Part II: Adjustments to Income

Line 11 - Educator expenses: $21

Line 12 - Certain business expenses of reservists, performing artists, and fee-basis government officials: $0

Line 13 - Health savings account deduction: $795

Line 14 - Moving expenses for members of the Armed Forces: $0

Line 15 - Deductible part of self-employment tax: $0

Line 16 - Self-employed SEP, SIMPLE, and qualified plans: $0

Line 17 - Self-employed health insurance deduction: $0

Line 18 - Penalty on early withdrawal of savings: $0 

Line 19 - Alimony paid: $0 

Line 20 - IRA deduction: $3,154

Line 21 - Student loan interest deduction: $1,663

Line 22 - Reserved for future use: $0

Line 23 - Archer MSA deduction: $0

Line 24 - Other adjustments: $390

Line 26 - Add lines 11 through 24. These are your adjustments to income. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 10: [__]

# Schedule 2 (Form 1040): Additional Taxes

## Part I: Tax

Line 1 - Alternative minimum tax: $3,052

Line 2 - Excess advance premium tax credit repayment. Attach Form 8962: $955

Line 3 - Add lines 1 and 2. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 17: [__]

## Part II: Other Taxes

Line 4 - Self-employment tax. Attach Schedule SE: [__]

Line 17 - Other additional taxes: $103

Line 21 - Add lines 4 and 17. These are your total other taxes. Enter here and on Form 1040 or 1040-SR, line 23, or Form 1040-NR, line 23b: [__]

# Schedule 3 (Form 1040): Additional Credits and Payments

# Part I: Nonrefundable Credits

Line 1 - Foreign tax credit: $133

Line 2 - Credit for child and dependent care expenses from Form 2441, line 11: $458

Line 3 - Education credits from Form 8863, line 19: $0

Line 4 - Retirement savings contributions credit: $1,170

Line 5a - Residential clean energy credit from Form 5695, line 15: $0

Line 5b - Energy efficient home improvement credit from Form 5695, line 32: $0

Line 6 - Other nonrefundable credits:

Line 6d - Elderly or disabled credit from Schedule R: $664

Line 6i - Qualified plug-in motor vehicle credit: $3,515

Line 6j - Alternative fuel vehicle refueling property credit: $280

Line 7 - Total other nonrefundable credits. Add lines 6d, 6i, and 6j: [__]

Line 8 - Add lines 1 through 4, 5a, 5b, and 7. Enter here and on Form 1040, 1040-SR, or 1040-NR, line 20: [__]

Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:
Let's go through the calculations step-by-step based on the provided information.

### Form 1040: U.S. Individual Income Tax Return

#### Basic Information
- Filing Status: Head of household
- Age: 68 (both John and his spouse)
- Itemized Deductions: False
- Dependents: 0

#### Income
- Line 1a: Total amount from Form(s) W-2, box 1: $129,500
- Line 1b: Household employee wages not reported on Form(s) W-2: $923
- Line 1c: Tip income not reported on line 1a: $2,144
- Line 1d: Nontaxable combat pay election: $210
- Line 1z: Add lines 1a through 1c: $129,500 + $923 + $2,144 = $132,567

- Line 2a: Tax-exempt interest: $1,318
- Line 2b: Taxable interest: $1,343
- Line 3a: Qualified dividends: $0
- Line 3b: Ordinary dividends: $1,650
- Line 4a: IRA distributions: $5,531
- Line 4b: Taxable amount: $1,023
- Line 5a: Pensions and annuities: $27,868
- Line 5b: Taxable amount: $2,679
- Line 6a: Social security benefits: $9,403
- Line 6b: Taxable amount: $1,176
- Line 7: Capital gain or (loss): $0
- Line 8: Additional income from Schedule 1, line 10: [__]

#### Schedule 1 (Form 1040): Additional Income and Adjustments to Income

Part I: Additional Income
- Line 1: Taxable refunds, credits, or offsets of state and local income taxes: $1,135
- Line 2a: Alimony received: $2,864
- Line 3: Business income or (loss): [__]
- Line 4: Other gains or (losses): $21,053
- Line 5: Rental real estate, royalties, partnerships, S corporations, trusts, etc.: $0
- Line 6: Farm income or (loss): $0
- Line 7: Unemployment compensation: $1,280
- Line 8: Other income: $337
- Line 10: Combine lines 1 through 8: $1,135 + $2,864 + $21,053 + $1,280 + $337 = $26,669

Part II: Adjustments to Income
- Line 11: Educator expenses: $21
- Line 12: Certain business expenses of reservists, performing artists, and fee-basis government officials: $0
- Line 13: Health savings account deduction: $795
- Line 14: Moving expenses for members of the Armed Forces: $0
- Line 15: Deductible part of self-employment tax: $0
- Line 16: Self-employed SEP, SIMPLE, and qualified plans: $0
- Line 17: Self-employed health insurance deduction: $0
- Line 18: Penalty on early withdrawal of savings: $0
- Line 19: Alimony paid: $0
- Line 20: IRA deduction: $3,154
- Line 21: Student loan interest deduction: $1,663
- Line 22: Reserved for future use: $0
- Line 23: Archer MSA deduction: $0
- Line 24: Other adjustments: $390
- Line 26: Add lines 11 through 24: $21 + $795 + $3,154 + $1,663 + $390 = $6,023

#### Form 1040: U.S. Individual Income Tax Return (Continued)

- Line 9: Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8: $132,567 + $1,343 + $1,650 + $1,023 + $2,679 + $1,176 + $0 + $26,669 = $167,107
- Line 10: Adjustments to income from Schedule 1, line 26: $6,023
- Line 11: Subtract line 10 from line 9: $167,107 - $6,023 = $161,084

#### Standard Deduction
- Line 12: Standard deduction for head of household (born before January 2, 1959): $22,650

- Line 13: Qualified business income deduction from Form 8995 or Form 8995-A: $6,727
- Line 14: Add lines 12 and 13: $22,650 + $6,727 = $29,377
- Line 15: Subtract line 14 from line 11: $161,084 - $29,377 = $131,707

#### Tax and Credits

Tax Calculation
- Line 16: Since the taxable income ($131,707) is over $100,000, we use the Tax Computation Worksheet for Head of Household.

Tax Computation Worksheet—Line 16 (Head of Household)
- At least $100,000 but not over $182,100
  - (a) Enter the amount from line 15: $131,707
  - (b) Multiplication amount: 24% (0.24)
  - (c) Multiply (a) by (b): $131,707 * 0.24 = $31,609.68
  - (d) Subtraction amount: $8,206.00
  - Tax: Subtract (d) from (c): $31,609.68 - $8,206.00 = $23,403.68

- Line 17: Amount from Schedule 2, line 3: $3,052 + $955 = $4,007
- Line 18: Add lines 16 and 17: $23,403.68 + $4,007 = $27,410.68

Credits
- Line 19: Child tax credit or credit for other dependents from Schedule 8812: $0
- Line 20: Amount from Schedule 3, line 8: $133 + $458 + $1,170 + $664 + $3,515 + $280 = $6,220
- Line 21: Add lines 19 and 20: $0 + $6,220 = $6,220
- Line 22: Subtract line 21 from line 18: $27,410.68 - $6,220 = $21,190.68

Other Taxes
- Line 23: Other taxes, including self-employment tax, from Schedule 2, line 21: $103
- Line 24: Add lines 22 and 23: $21,190.68 + $103 = $21,293.68

#### Payments
- Line 25: Federal income tax withheld: $5,512
- Line 26: 2023 estimated tax payments and amount applied from 2022 return: $0
- Line 27: Earned income credit (EIC): $0
- Line 28: Additional child tax credit from Schedule 8812: $0
- Line 29: American opportunity credit from Form 8863, line 8: $0
- Line 30: Reserved for future use: $0
- Line 31: Amount from Schedule 3, line 15: [__]
- Line 32: Add lines 27, 28, 29, and 31: $0
- Line 33: Add lines 25, 26, and 32: $5,512

#### Amount You Owe
- Line 37: Subtract line 33 from line 24: $21,293.68 - $5,512 = $15,781.68

### Final Calculation
The total tax owed is $15,781.68.
</example>
"""

example_dict = {
    "gpt-4o-2024-08-06": example_gpt,
    "gpt-4o-mini": example_gpt,
    "gpt-4o": example_gpt,
    "o1-preview": example_claude,
    "claude-3-5-sonnet-20241022": example_claude,
    "qwen2.5-72b-instruct": example_qwen,
    "meta/llama-3.1-405b-instruct-maas": example_llama,
    "meta/llama-3.1-70b-instruct-maas": example_llama,
}
