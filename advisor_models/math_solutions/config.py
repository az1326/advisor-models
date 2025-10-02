"""Configuration for math solutions domain.

Contains system prompts, student personas, and reward functions for math solution writing.
Defines different student preferences for solution style and evaluation criteria.
"""

# Students with different solution style preferences (2x2 quadrant design)
STUDENTS = ["Alex", "Blake", "Casey", "Dana"]

# Student personas based on two binary axes:
# Axis 1: Multiple solution methods (yes/no)
# Axis 2: Asks user questions when presenting (yes/no)
STUDENT_PERSONAS = {
    "Alex": {
        "style": "Multiple methods, with questions",
        "description": "Wants to see multiple solution approaches and likes when the tutor asks questions during the solution to check understanding or guide thinking.",
        "judge_criteria": "Criteria 1: Shows or mentions multiple approaches towards solving the problem.\nCriteria 2: Poses questions to the reader at any point in the solution.",
    },
    "Blake": {
        "style": "Multiple methods, no questions",
        "description": "Wants to see multiple solution approaches but prefers direct presentation without interactive questions or prompts.",
        "judge_criteria": "Criteria 1: Shows or mentions multiple approaches towards solving the problem.\nCriteria 2: Does not pose any question to the reader at any point in the solution.",
    },
    "Casey": {
        "style": "Single method, with questions",
        "description": "Prefers one clear solution method but likes when the tutor asks questions during the solution to check understanding or guide thinking.",
        "judge_criteria": "Criteria 1: Presents and mentions only a single approach towards solving the problem.\nCriteria 2: Poses questions to the reader at any point in the solution.",
    },
    "Dana": {
        "style": "Single method, no questions",
        "description": "Prefers one clear solution method with direct presentation without interactive questions or prompts.",
        "judge_criteria": "Criteria 1: Presents and mentions only a single approach towards solving the problem.\nCriteria 2: Does not pose any question to the reader at any point in the solution.",
    },
}

# Advisor model prompts
ADVISOR_SYSTEM_PROMPT = """You are an advisor to a math tutor. Provide specific guidance to the math tutor on how to present a solution to a math problem in a way that matches the student's preferred learning style. For example, you might provide guidance on whether to include multiple solution methods, to ask questions of the student during the solution, to provide very detailed or big-picture explanations, and/or to use visual guides. You can provide guidance on multiple aspects at once. Consider the student's preferences and adapt your advice accordingly."""

ADVISOR_INSTRUCTION = """You need to provide advice to a tutor presenting a solution to a math problem to {student}. For example, you might provide guidance on whether to include multiple solution methods, to ask questions of the student during the solution, to provide very detailed or big-picture explanations, and/or to use visual guides. You can provide guidance on multiple aspects at once.

The math problem is: {problem}

Provide specific advice about how to present a solution to this problem so that the solution aligns with the student {student}'s learning style. Keep your advice to 3-4 sentences and focus on providing advice on the structure and style of the solution, not the content of the actual solution."""

# Student model prompts
STUDENT_SYSTEM_PROMPT = """You are a math tutor tasked with writing a solution to a math problem that will help a student learn. You will receive advice from an advisor who knows the student's learning preferences. Based on the problem and advisor guidance, write a complete solution that follows the guidance provided. Always end your solution with the final answer in \\boxed{} format."""

STUDENT_INSTRUCTION = """Math Problem: {problem}

Advisor Guidance:
{advisor_feedback}

Write a solution for this math problem following the advisor's guidance on how to write a solution. Your solution must implement all the advisor's guidance. End with the final answer in \\boxed{{}} format."""

# Judge prompts
STYLE_JUDGE_SYSTEM_PROMPT = "You are an evaluator of educational material. Your task is to evaluate whether the given solution aligns with the given criteria."

STYLE_JUDGE_PROMPT = """You are evaluating whether a math solution aligns with the specified evaluation criteria.

Solution to evaluate:
{solution}

Evaluation criteria:
{judge_criteria}

If the solution aligns with both evaluation criteria, respond with "ACCEPT".
If the solution aligns with only one evaluation criteria, respond with "PARTIAL".
If the solution does not align with either evaluation criteria, respond with "REJECT".

Briefly reason about whether each criteria is met before responding."""
