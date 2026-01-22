
# basic system summary prompt

SUMMARY_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a memory compression engine. "
        "Only extract facts explicitly stated by the user. "
        "DO NOT infer, guess, or assume. "
        "If no new facts exist, return the old summary unchanged."
        "If newer information contradicts older summary facts, "
        "replace the old fact with the new one"
    )
}






# build the summry from the conversation

def build_summary_prompt(old_summary, conversation_chunk):
    
    return [
        SUMMARY_SYSTEM_PROMPT,        
        {
            "role": "user",
            "content": f"""
OLD SUMMARY:
{old_summary}

NEW CONVERSATION:
{conversation_chunk}
"""
        }
    ]





def should_summarize(conversation, max_turns=8):
    user_turns = [
        m for m in conversation
        if m["role"] in ("user", "assistant")
    ]
    return len(user_turns) > max_turns








# limit the summary size
def clamp_summary(summary, SUMMARY_MAX_CHARS=500):
    return summary[:SUMMARY_MAX_CHARS]







# extract system message so that we can't summarize that by mistakes
def extract_user_assistant_messages(conversation):
    return [
        m for m in conversation
        if m["role"] in ("user", "assistant")
    ]






def message_to_text(messages):
    return "\n".join(
        f"{m['role']}: {m['content']}" for m in messages
    )






def update_summary(old_summary, conversation_chunk):
    chunk_text = message_to_text(conversation_chunk)
    messages = build_summary_prompt(old_summary, chunk_text)
    return fake_summarizer(messages)






def fake_summarizer(messages):
    return "Summary of key facts and goals so far."






def is_memory_safe(text: str) -> bool:
    forbidden = ["remember", "store this", "save this", "add to memory"]
    return not any(f in text.lower() for f in forbidden)
