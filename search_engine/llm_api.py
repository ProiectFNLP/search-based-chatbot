from openai import OpenAI
from ollama import chat
from settings import settings
import requests
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

def get_openai_client():
    if settings.openai_api_key is None:
        raise RuntimeError("OpenAI API key not configured")

    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value()
    )

#client = OpenAI(api_key=settings.openai_api_key.get_secret_value())



# def generate_response(context_information:  Generator[str, None, None], prompt: str) -> str:

#     messages = [
#         {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
#         {"role": "user", "content": GENERATION_USER_PROMPT.format(context_information=context_information, query=prompt)}
#     ]

#     response = send_request(messages)
#     return response

LLAMA_MODEL = "llama3.2"
LLAMA_URL = "http://localhost:11434/api/generate"

from ollama import generate

def generate_response(context_results, prompt):
    # Concatenează contextul din document
    context_text = "\n".join(context_results)

    # Construiește textul complet trimis la LLaMA
    full_prompt = f"Context trimis la LLaMA:\n{context_text}\n\nIntrebare: {prompt}\nRaspuns:"
    print(full_prompt)
    # Generează răspuns folosind modelul local LLaMA
    result = generate(LLAMA_MODEL, full_prompt)

    # Returnează doar textul generat
    response = result.response
    return response

def generate_summary(prompt: str, answer: str, conversation_summary: str) -> str:

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_USER_PROMPT.format(query=prompt, answer=answer, conversation_summary=conversation_summary)}
    ]

    response = send_request(messages)
    return response


def send_request(messages: list[dict[str, str]]) -> str:
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=messages
    # )
    response = chat(
        model="llama3.2",
        messages=messages
    )
    return response["message"]["content"]
    #return response.choices[0].message.content

def call_llama(prompt: str) -> str:
    """
    Trimite promptul la modelul LLaMA local prin Ollama API
    si întoarce raspunsul ca string.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",  # numele modelului tău din ollama list
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except requests.exceptions.RequestException as e:
        return f"Error calling LLaMA: {e}"
