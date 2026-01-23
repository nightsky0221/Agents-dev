def summarize_conversation(conversation):
    """
    Create a short memory summary from the conversation.
    """

    text = []
    for msg in conversation:
        if msg["role"] in ("user", "assistant"):
            text.append(msg["content"])

    joined = "\n".join(text)

    return {
        "role": "system",
        "content": f"Conversation summary:\n {joined[-1500:]}"
    }