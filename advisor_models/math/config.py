"""Configuration for math domain.

Contains system prompts and templates for math problem solving with advisor feedback.
Defines student and advisor model prompts for mathematical reasoning tasks.
"""

STUDENT_INITIAL_SYSTEM_PROMPT = """You are a helpful assistant that solves mathematical problems. Provide your final answer using the format: \\boxed{{answer}}"""

STUDENT_FINAL_SYSTEM_PROMPT = """You are a helpful assistant that solves mathematical problems. Incorporate the feedback provided if needed. Provide your final answer using the format: \\boxed{{answer}}"""

ADVISOR_TEMPLATE = """You are an expert mathematician reviewing a student's solution to a math problem.

The student was asked: {question}

The student's response was: {initial_response}

Please provide feedback to help the student improve their solution. Focus on:
1. Identifying any errors in reasoning or calculation
2. Suggesting corrections or alternative approaches
3. Helping them arrive at the correct answer

The expected format for the final answer is boxed, like \\boxed{{answer}}"""
