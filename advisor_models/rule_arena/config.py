"""Configuration for rule arena (US tax) domain.

Contains system prompts and templates for US tax calculation with advisor feedback.
"""

STUDENT_SYSTEM_PROMPT = "You are a helpful US taxation consultant. End your response with: '1. The total tax owed is $xxx.' (xxx is a number) if there is tax owed. 2. The total tax overpaid is $xxx.' (xxx is a number) if there is tax overpaid (and should be refunded)."

ADVISOR_INSTRUCTIONS = "You are an advisor whose job is to review the solution to identify flaws and provide specific feedback to improve it if needed. Focus on accuracy and completeness."

PROBLEM_TEMPLATE = """You are given several forms used to report US income tax and the instructions or rules about how to fill the forms. Then you will be given the income and/or payment information about a tax payer According to the given information. You should calculate the income tax owed by this payer.

IRS Forms for the tax payer:
$forms
Calculate the tax owed by the payer step-by-step according to the information provided by the forms. You should calculate all fields marked with [__]. DO NOT round numbers without explicit instructions. End your response with:
1. "The total tax owed is $xxx." (xxx is a number) if there is tax owed.
2. "The total tax overpaid is $xxx." (xxx is a number) if there is tax overpaid (and should be refunded).
Your response:"""
