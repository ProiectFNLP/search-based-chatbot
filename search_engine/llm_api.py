from openai import OpenAI
import json
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


# Global variables for model caching in worker processes
_model_cache = {}
_tokenizer_cache = {}


def _load_model_worker(model_path: str):
    """
    Load the Flan-T5 model in a worker process.
    This function is called in the worker process to cache the model.
    """
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    
    if model_path not in _model_cache:
        print(f"Loading Flan-T5 model from: {model_path}")
        _tokenizer_cache[model_path] = AutoTokenizer.from_pretrained(model_path)
        _model_cache[model_path] = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        print(f"Model loaded successfully")
    
    return _model_cache[model_path], _tokenizer_cache[model_path]


def _generate_with_flan_t5(model_path: str, prompt_text: str) -> str:
    """
    Generate response using Flan-T5 model.
    This function runs in a worker process.
    """
    model, tokenizer = _load_model_worker(model_path)
    
    # Tokenize input
    inputs = tokenizer(prompt_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate
    outputs = model.generate(
        inputs["input_ids"],
        max_length=512,
        num_beams=4,
        early_stopping=True,
        do_sample=False
    )
    
    # Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("resposne from flan-t5-base: ", response)
    return response


def _extract_context_from_generator(context_information: Generator[str, None, None]) -> str:
    """
    Extract and format context from the search results generator.
    The generator yields JSON-formatted strings like: "data: {...}\n\n"
    """
    context_parts = []
    
    for item in context_information:
        # Parse the JSON string (format: "data: {...}\n\n")
        if item.startswith("data: "):
            json_str = item[6:].strip()  # Remove "data: " prefix
            try:
                data = json.loads(json_str)
                if "results" in data:
                    for result in data["results"]:
                        if "paragraph" in result and result["paragraph"]:
                            paragraph = result["paragraph"]
                            if isinstance(paragraph, bytes):
                                paragraph = paragraph.decode('utf-8')
                            context_parts.append(paragraph)
            except json.JSONDecodeError:
                continue
    
    # Combine all paragraphs into a single context string
    context_text = "\n\n".join(context_parts)
    return context_text


def generate_response_local(context_information: Generator[str, None, None], prompt: str) -> str:
    """
    Generate response using local Flan-T5 model.
    This function should be called from an async context using run_in_executor.
    """
    # Determine model path
    model_path = settings.flan_t5_model_path or "google/flan-t5-base"
    
    # Extract context from generator
    context_text = _extract_context_from_generator(context_information)
    
    # Format prompt for Flan-T5 (text-to-text model, not chat-based)
    # Combine system and user prompts into a single text input
    full_prompt = f"""{GENERATION_SYSTEM_PROMPT.strip()}

User query: {prompt}

Retrieved context:
{context_text}

Using only the retrieved context, produce the best possible answer to the user query.
Include in your answer the specific context snippets you are basing your response on, quoted in double quotes ""."""
    
    # Run model inference (this will be called in a worker process)
    print("full_prompt: ", full_prompt)
    response = _generate_with_flan_t5(model_path, full_prompt)
    print("response from flan-t5-base: ", response)
    return response

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
