
# define LLMs model that generates responses to the requests

def fake_llm(messages, persona):

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
        return(
                # "- I can help with NLP and LLM topics only.\n"
                # "- This question is outside my scope.\n"
                # "- Please ask another NLP-related question.\n"
            "\n".join(bullets[:5]) +
            "\nWhat part of tokenization would you like to explore next?"
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
    
    elif persona == "other":
        return(
            "Thanks for your question. "
            "This request doesn't fall under NLP tutoring or product support. "
            "Could you please clarify or provide more details so I can help you."
        )
