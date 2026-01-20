import json

# define LLMs model that generates responses to the requests

def llm_call(messages, persona):
     
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
            "answer": "This is a structured answer to your question.",
            "confidence": 0.2
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

MAX_RETRIES = 3

class LLMError(Exception):
    pass

def call_llm_with_retries(messages, call_fn, persona):
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = call_fn(messages, persona)
            return raw
        except Exception as e:
            last_error = e
            print(f"[LLM] Attempt {attempt} failed: {e}")

    raise LLMError("LLM failed after max retries") from last_error
