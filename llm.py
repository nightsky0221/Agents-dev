import json
import time


LLM_CONFIG = {
    "temperature": 0.0,
    "top": 1.0,
    "max_tokens": 512,
}



def llm_call(messages, persona):

    return generate_llm_response(messages, config=LLM_CONFIG)

    MAX_TOKENS_PER_CALL = 500
    estimated = estimate_token(messages)
    if estimated > MAX_TOKENS_PER_CALL:
        raise RuntimeError(
            f"Token budget exceeded: {estimated} > {MAX_TOKENS_PER_CALL}"
        )

    # Detect JSON-enforced mode
    json_mode = any(
        isinstance(m, dict)
        and m.get("role") == "system"
        and isinstance(m.get("content"), str)
        and "valid JSON" in m["content"]
        for m in messages
    )

    # Test

    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            print(f"[DEBUG] Invalid message at index {i}: {m}")

    if json_mode:
        return json.dumps({
            "answer": "I will calculate this using a calculator",
            "confidence": 0.9,
            "tool_request": {
                "tool": "calculator",
                "arguments": {
                    "expression": "23 * 17"
                }
            }
        })

    # user_input is used for comparing the request with keywords, it's not currently activated but you will use it if neccessary
    user_input = messages[-1]["content"]

    if persona == "tutor":
        # nlp_keywords = [
        #     "nlp", "llm", "token", "tokenization", "embedding", "transformer", "attention", "language model"
        # ]
        bullets = [
                "- Tokenization splits text into smaller units called tokens.",
                "- Tokens can be words, subwords, or characters.",
                "- Models process tokens instead of raw text.",
                "- Tokenization affects model vocabulary and performance.",
                "- Different models use different tokenizers."
        ]
        # if not any(k in user_input for k in nlp_keywords):
        return (
                # "- I can help with NLP and LLM topics only.\n"
                # "- This question is outside my scope.\n"
                # "- Please ask another NLP-related question.\n"
            "\n".join(bullets[:5]) +
            "\nWhat part of tokenization would you like to explore next?"
            # {"answer": "hi", "confidence": 0.2}
        )
        
        # return "Tutor-style response(step-by-step)."
    
    elif persona == "support":
        
        # issue_keywords = [
        #     "crash", "error", "bug", "login", "issue", "problem"
        # ]
        # if not any(k in user_input for k in issue_keywords):
            # return(
            #     "Thanks for reaching out. "
            #     "Could you please provide more details about the issue "
            #     "so I can assist you further?"
            # )
        
        return(
            "Sorry you are experiencing this issue. "
            "Please try clearing your app cache and restarting the app. "
            "If the problem persists, I will escalate this to our technical team."
        )
    
    return(
        "Thanks for your question. "
        "This request doesn't fall under NLP tutoring or product support. "
        "Could you please clarify or provide more details?"
    )








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
            
            if attempt == 0:
                messages.append({
                    "role": "system",
                    "content": FORMAT_CORRECTION
                })
            continue
        
    return {
        "answer": "I couldn't generate a reliable response for that request.",
        "confidence": 0.0,
        "tool_request": None,
        **make_error(
            code="LLM_RETRY_EXAUSTED",
            message="Model failed to produce valid output after retries",
            retryable=False,
        )
    }






def retry_backoff(attempt: int):
    time.sleep(0.5 * (attempt +1))







def generate_llm_response(messsages, config):
    """
    This is the ONLY place allowed to talk to an LLM.
    Can be mocked, local, or remote.
    """
    return {
        "role": "assistant",
        "content": "Mock response (LLM not wired yet)",
    }







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