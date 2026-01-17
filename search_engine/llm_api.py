from openai import OpenAI
import json
from settings import settings
import requests
from typing import Generator
from src.constants import TOP_K

# Optional ollama import - only needed for llama model
try:
    from ollama import chat, generate
    OLLAMA_AVAILABLE = True
    print("✓ Ollama package is available")
except ImportError:
    chat = None
    generate = None
    OLLAMA_AVAILABLE = False
    print("⚠ WARNING: Ollama package is not installed. Llama model will not be available.")
    print("  Install it with: pip install ollama")

GENERATION_SYSTEM_PROMPT = """
    You are a retrieval-augmented generation (RAG) assistant.

    Answer the user question using the information present in the retrieved context.
    Do NOT add external knowledge or assumptions.

    Write the answer in your own words, in a clear and concise manner.
    Do NOT copy full sentences verbatim from the context.

    After the answer, provide a short list of quoted context snippets that support it.
    If the context is insufficient, explicitly say so.
    Be concise and factual.
    """



GENERATION_USER_PROMPT = """
    User question:
    {query}

    Retrieved context:
    {context_information}
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

LLAMA_MODEL = "llama3.2"
LLAMA_URL = "http://localhost:11434/api/generate"

def generate_response(context_results, prompt, model):
    # Extract context for all models
    context_list = list(context_results)
    context_text = "\n".join(context_list[:3])

    if model == 'llama':
        if not OLLAMA_AVAILABLE or generate is None:
            print("❌ ERROR: Ollama is not available. Cannot generate response with Llama model.")
            print("  Please install ollama: pip install ollama")
            print("  And ensure Ollama service is running: ollama serve")
            return "Error: Ollama is not available. Please install ollama package and start the Ollama service."

        # Construiește textul complet trimis la LLaMA
        full_prompt = GENERATION_USER_PROMPT.format(query=prompt, context_information=context_text)
        print(full_prompt)
        # Generează răspuns folosind modelul local LLaMA
        try:
            result = generate(LLAMA_MODEL, full_prompt)
            # Returnează doar textul generat
            response = result.response
            return response
        except Exception as e:
            print(f"❌ ERROR: Failed to generate response with Ollama: {e}")
            print("  Make sure Ollama service is running: ollama serve")
            print("  And the model is available: ollama pull llama3.2")
            return f"Error generating response: {str(e)}"
    elif model == 'gpt-5-nano':
        messages = [
            {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": GENERATION_USER_PROMPT.format(context_information=context_text, query=prompt)}
        ]
        return send_request(messages, model)
    elif model == 'flan-t5-base':
        # Use local Flan-T5
        return generate_response_local(context_results, prompt)
    elif model == 'qwen2-1.5b':
        # Use local Qwen2-1.5B-Instruct
        return generate_response_local_qwen(context_results, prompt)
    else:
        return f"Unsupported model: {model}"


def generate_summary(prompt: str, answer: str, conversation_summary: str) -> str:

    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARY_USER_PROMPT.format(query=prompt, answer=answer, conversation_summary=conversation_summary)}
    ]

    response = send_request(messages)
    return response

def send_request(messages: list[dict[str, str]], model='llama') -> str:
    if model == 'llama':
        if not OLLAMA_AVAILABLE or chat is None:
            print("❌ ERROR: Ollama is not available. Cannot send request with Llama model.")
            print("  Please install ollama: pip install ollama")
            print("  And ensure Ollama service is running: ollama serve")
            return "Error: Ollama is not available. Please install ollama package and start the Ollama service."

        try:
            response = chat(
                model="llama3.2",
                messages=messages
            )
            return response["message"]["content"]
        except Exception as e:
            print(f"❌ ERROR: Failed to send request to Ollama: {e}")
            print("  Make sure Ollama service is running: ollama serve")
            print("  And the model is available: ollama pull llama3.2")
            return f"Error sending request: {str(e)}"
    elif model == 'gpt-5-nano':
        client = get_openai_client()
        try:
            response = client.chat.completions.create(
                model="gpt-5-nano",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"❌ ERROR: Failed to send request to OpenAI: {e}")
            return f"Error sending request: {str(e)}"


# Global variables for model caching in worker processes
_model_cache = {}
_tokenizer_cache = {}

# Global variables for Qwen2 model caching in worker processes
_model_cache_qwen = {}
_tokenizer_cache_qwen = {}


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
    print("Response from flan-t5-base: ", response)
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

    context_parts = context_parts[:TOP_K]

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
    generation_user_prompt = GENERATION_USER_PROMPT.format(context_information=context_text, query=prompt)

    full_prompt = generation_user_prompt + "\n\n" + GENERATION_SYSTEM_PROMPT

    # Run model inference (this will be called in a worker process)
    print("full_prompt: ", full_prompt)
    response = _generate_with_flan_t5(model_path, full_prompt)
    return response


def _load_model_worker_qwen(model_path: str):
    """
    Load the Qwen2-1.5B-Instruct model in a worker process.
    This function is called in the worker process to cache the model.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if model_path not in _model_cache_qwen:
        print(f"Loading Qwen2 model from: {model_path}")
        _tokenizer_cache_qwen[model_path] = AutoTokenizer.from_pretrained(model_path)
        _model_cache_qwen[model_path] = AutoModelForCausalLM.from_pretrained(model_path)
        # Set pad_token if not present
        if _tokenizer_cache_qwen[model_path].pad_token is None:
            _tokenizer_cache_qwen[model_path].pad_token = _tokenizer_cache_qwen[model_path].eos_token
        print(f"Qwen2 model loaded successfully")

    return _model_cache_qwen[model_path], _tokenizer_cache_qwen[model_path]


def _generate_with_qwen(model_path: str, messages: list[dict[str, str]]) -> str:
    """
    Generate response using Qwen2-1.5B-Instruct model.
    This function runs in a worker process.
    """
    model, tokenizer = _load_model_worker_qwen(model_path)

    # Apply chat template to format messages
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True
    )

    # Generate
    outputs = model.generate(
        inputs["input_ids"],
        num_beams=4,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode output (skip the input prompt)
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print("Response from qwen2-1.5b: ", response)
    return response


def generate_response_local_qwen(context_information: Generator[str, None, None], prompt: str) -> str:
    """
    Generate response using local Qwen2-1.5B-Instruct model.
    This function should be called from an async context using run_in_executor.
    """
    # Determine model path
    model_path = settings.qwen2_model_path or "Qwen/Qwen2-1.5B-Instruct"

    # Extract context from generator
    context_text = _extract_context_from_generator(context_information)

    # Format messages for Qwen2 (chat-based model)
    messages = [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": GENERATION_USER_PROMPT.format(context_information=context_text, query=prompt)}
    ]

    # Run model inference (this will be called in a worker process)
    print("Messages for qwen2-1.5b: ", messages)
    response = _generate_with_qwen(model_path, messages)
    return response
