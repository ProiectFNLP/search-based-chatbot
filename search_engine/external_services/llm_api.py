from typing import Generator

import openai

GENERATION_SYSTEM_PROMPT = """ """
GENERATION_USER_PROMPT = """ """

SUMMARY_SYSTEM_PROMPT = """ """
SUMMARY_USER_PROMPT = """ """


def generate_response(context_information:  Generator[str, None, None], prompt: str) -> str:

    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": GENERATION_USER_PROMPT.format(context_information=context_information, prompt=prompt)}
    ]

    response = send_request(messages)
    return response


def generate_summary(prompt: str, response: str, conversation_summary: str) -> str:

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_USER_PROMPT.format(prompt=prompt, response=response, conversation_summary=conversation_summary)}
    ]

    response = send_request(messages)
    return response


def send_request(messages: list[dict[str, str]]) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=messages
    )

    return response.choices[0].message.content
