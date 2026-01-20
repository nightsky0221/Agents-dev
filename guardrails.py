INJECTION_PATTERNS = [
    "ignore previous",
    "ignore all instructions",
    "disregard system",
    "you are now",
    "change your role",
    "forget all rules",
    "output yaml",
    "output xml",
    "add this to memory",
    "remember that",
]

def is_prompt_injection(user_input: str) -> bool:
    text = user_input.lower()
    return any(p in text for p in INJECTION_PATTERNS)

def guard_input(user_input: str) -> str:
    if is_prompt_injection(user_input):
        raise ValueError("Prompt injection detected")
    return user_input