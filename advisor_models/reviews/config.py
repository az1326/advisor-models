"""Configuration for reviews domain.

Contains system prompts, user personas, and reward functions for review writing.
Defines length and reading level preferences for different user types.
"""

# People with different length preferences
LENGTH_PEOPLE = ["Alan", "Parth", "Matei", "Joey", "Alex"]
LEVEL_PEOPLE = ["Maya", "Rafael", "Aiden", "Leila", "Jonah"]

# Preferred word counts for each person
LENGTH_CRITERIA = {
    "Alan": 500,
    "Parth": 50,
    "Matei": 10,
    "Joey": 200,
    "Alex": 1000,
}

LEVEL_CRITERIA = {
    "Maya": "Is the following review appropriate for an elementary school student in terms of reading level? If the review is too complex or too simple for an elementary school student, respond with No.\n\n{review}\n\nRespond only with Yes or No.",
    "Rafael": "Is the following review appropriate for a middle school student in terms of reading level? If the review is too complex or too simple for a middle school student, respond with No.\n\n{review}\n\nRespond only with Yes or No.",
    "Aiden": "Is the following review appropriate for a high school student in terms of reading level? If the review is too complex or too simple for a high school student, respond with No.\n\n{review}\n\nRespond only with Yes or No.",
    "Leila": "Is the following review appropriate for a college student in terms of reading level? If the review is too complex or too simple for a college student, respond with No.\n\n{review}\n\nRespond only with Yes or No.",
    "Jonah": "Is the following review appropriate for a college professor in terms of reading level? If the review is too simple for a college professor, respond with No.\n\n{review}\n\nRespond only with Yes or No.",
}

# System prompt for advisor model
LENGTH_ADVISOR_SYSTEM_PROMPT = """You are a review writing advisor. Provide specific guidance for writing a review that matches the person's preferences. Consider the length preferences and style that would work best for the target person."""

LEVEL_ADVISOR_SYSTEM_PROMPT = """You are a review writing advisor. Provide specific guidance for writing a review that matches the person's preferences. Consider the reading level of the target person."""

WEAK_ADVISOR_SYSTEM_PROMPT = """You are a review writing advisor. Provide specific guidance for writing a review."""


# Instructions for advisor model
LENGTH_ADVISOR_INSTRUCTION = """You need to provide advice for writing a review for {person}.

The task is: {prompt}

Provide specific advice about the review that would work best for {person}. Think carefully about the length of the review in your advice. Keep your advice to 3-4 sentences."""

LEVEL_ADVISOR_INSTRUCTION = """You need to provide advice for writing a review for {person}.

The task is: {prompt}

Provide specific advice about the review for {person}. Think carefully about the reading level of the target person in your advice. Keep your advice to 3-4 sentences."""

WEAK_ADVISOR_INSTRUCTION = """You need to provide advice for writing a review for {person}.

The task is: {prompt}

Provide specific advice about the review for {person}. Keep your advice to 3-4 sentences."""


# System prompt for student model (writes the actual review)
STUDENT_SYSTEM_PROMPT = """You are a review writer. Based on the prompt and advisor guidance, write a review that follows the guidance provided. Write a clear, well-structured review."""

STUDENT_INSTRUCTION = """Review Prompt: {prompt}

Advisor Guidance:
{advisor_feedback}

Write a review following the advisor's guidance."""


def compute_length_reward(review_text: str, preferred_length: int) -> float:
    """Compute reward based on how well the review matches the person's length preference."""
    word_count = len(review_text.split())

    distance = abs(word_count - preferred_length)

    # Inverse distance reward - never reaches 0, smooth gradient for learning
    # Reward approaches 1.0 as distance approaches 0
    reward = 1.0 / (1.0 + distance / preferred_length)

    return reward
