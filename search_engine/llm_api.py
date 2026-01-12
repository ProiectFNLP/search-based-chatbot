from openai import OpenAI

from settings import settings

from typing import Generator

GENERATION_SYSTEM_PROMPT = """
    You are a retrieval-augmented generation (RAG) assistant.
    Your task is to answer user questions using only the provided contextual information.
    If the context does not contain enough information to answer the question, state that explicitly.
    Do not invent facts or rely on external knowledge.
    Respond clearly and concisely.
    Always indicate the context snippets you used in your answer by quoting them (e.g., in double quotes "").
    """


GENERATION_USER_PROMPT = """
    User query:
    {query}

    Retrieved context:
    {context_information}

    Using only the retrieved context, produce the best possible answer to the user query.
    Include in your answer the specific context snippets you are basing your response on, quoted in double quotes "".
    """


SUMMARY_SYSTEM_PROMPT = """
    You are a conversation summarization assistant.
    Your task is to maintain a concise, up-to-date summary of a conversation between a user and an assistant.
    The summary should preserve important facts, decisions, and context needed for future turns.
    Avoid redundancy and remove obsolete or repeated information.
    """

SUMMARY_USER_PROMPT = """
    Existing conversation summary:
    {conversation_summary}

    Latest exchange:
    - User query: {query}
    - Assistant response: {answer}

    Update the conversation summary by integrating the new exchange.
    Rewrite the summary if necessary to keep it concise, non-redundant, and accurate.
    Return only the updated summary.
    """


client = OpenAI(api_key=settings.openai_api_key.get_secret_value())



def generate_response(context_information:  Generator[str, None, None], prompt: str) -> str:

    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": GENERATION_USER_PROMPT.format(context_information=context_information, query=prompt)}
    ]

    response = send_request(messages)
    return response


def generate_summary(prompt: str, answer: str, conversation_summary: str) -> str:

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_USER_PROMPT.format(query=prompt, answer=answer, conversation_summary=conversation_summary)}
    ]

    response = send_request(messages)
    return response


def send_request(messages: list[dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    return response.choices[0].message.content
