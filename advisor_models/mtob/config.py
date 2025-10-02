"""Configuration for MTOB (Machine Translation from One Book) domain.

Contains system prompts and instructions for machine translation with advisor feedback.
Kalamang->English translation tasks.
"""

ADVISOR_INITIAL_INSTRUCTIONS = "Kalamang is a language spoken on the Karas Islands in West Papua. You are an advisor tasked with helping a model translate the following sentence from Kalamang to English: {source_text}"

ADVISOR_FINAL_INSTRUCTIONS = "\nNow write the advice. You must not provide a full translation in your advice; your advice should be helpful for a student wishing to do the translation on their own and learn from the process.\nKalamang: {source_text}"

STUDENT_INSTRUCTIONS = """Kalamang is a language spoken on the Karas Islands in West Papua. Translate the following sentence from Kalamang to English: {original_question}
Here is some additional reference material:
{reference_materials}

Here is some advice from the advisor: {advisor_feedback}

Now determine the translation.
Kalamang: {original_question}
Reason over the information and end your response with 'Translation: {{translation}}'"""
