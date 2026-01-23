import time
import requests

DEBUG_LLM = False # set False for silent mode
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL_NAME = "llama3"

def llm_call(messages, persona):

    final_message = []

    if persona:
        final_message.append(persona)
    
    final_message.extend(messages)

    payload = {
        "model": MODEL_NAME,
        "messages": final_message,
        "stream": False
    }

    if DEBUG_LLM:
        print("\n===== LLM PROMPT =====")
        for m in messages:
            print(f"{m['role'].upper()}: {m['content']}")
        print("======================\n")

    response = requests.post(OLLAMA_URL, json=payload)
    
    raw = response.json()["message"]["content"]

    if DEBUG_LLM:
        print("\n===== LLM RAW OUTPUT =====")
        print(raw)
        print("==========================\n")

    return raw

LLM_CONFIG = {
    "temperature": 0.0,
    "top": 1.0,
    "max_tokens": 512,
}

def estimate_token(messages: list[dict]) -> int:
    """
    rough token estimator
    1 token ~ 4 characters (safe approximation).
    """
    total_chars = 0
    for msg in messages:
        total_chars += len(msg.get("content", ""))
    return total_chars // 4

FORMAT_CORRECTION = """
Your previous response did not match the required JSON schema.
Return ONLY valid JSON.
Do not include explanations, markdown, or extra text.
"""

class LLMError(Exception):
    pass

def call_llm_with_retries(messages, call_fn, persona):
    MAX_RETRIES = 3

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_fn(messages, persona)
            return raw
        
        except ValueError as e:
            log_event("llm_retry",{
            "attempt": attempt,
            "reason": "invalid_json",
            "error": str(e)
            })
            
            if attempt == 1:
                messages.append({
                    "role": "system",
                    "content": FORMAT_CORRECTION
                })
            continue
        
    return {
        "type": "chat",
        "answer": "I encountered an internal error but recovered safely.",
        "confidence": 0.1,
        "tool_request": None,
    }

def retry_backoff(attempt: int):
    time.sleep(0.5 * (attempt +1))

def make_error(code: str, message: str, retryable: bool = False):
    return{
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    }

def log_event(event_type: str, payload: dict):
    print(f"[{event_type}] {payload}")