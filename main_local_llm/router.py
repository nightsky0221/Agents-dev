
# define the persona router so that we can select the right persona that fits the user input well

def route_persona(user_input):

    text = user_input.lower()

# Set keywords for each persona

    support_keywords = [
        "crash", "error", "bug", "login", "issue", "problem", "doesn't work", "failed", "help", "support"
    ]

    tutor_keywords = [
        "explain", "what is", "how does", "token", "embedding", "llm", "nlp", "transformer", "attention", "model"
    ]

# decide which persona is responsible for current user request by comparing it with keywords.

    if any(k in text for k in support_keywords):
        return "support"
    elif any(k in text for k in tutor_keywords):
        return "tutor"
    else:
        return "other"
