import llm
import Json_structure as js
import tools

def single_step_decision(messages, persona):
    llm_response = llm.call_llm_with_retries(
        messages=messages,
        persona=persona,
        call_fn=llm.llm_call
    )

    return {
        "type": "chat",
        "answer": llm_response,
        "confidence": 0.8,
        "tool_request": None
    }

def build_messages_json(persona_prompt, conversation):
    messages = [persona_prompt]

    messages.extend([
        {"role": m["role"], "content": m["content"]}
        for m in conversation
    ])
    return messages


def build_messages_loop(conversation):
    messages = [js.JSON_SYSTEM_PROMPT]

    if conversation and conversation[0]["role"] == "system":
        messages.append(conversation[0])
        rest = conversation[1:]
    else:
        rest = conversation

    messages.extend([
        {"role": m["role"], "content": m["content"]}
        for m in rest
    ])
    return messages

def execute_tool_from_parsed(parsed, conversation):
    tool_request = parsed.get("tool_request")

    if not tool_request:
        return conversation
    
    tool_name, arguments = tools.validate_tool_request(tool_request)
    observation = tools.execute_tool(tool_name, arguments)

    conversation.append({
        "role": "tool",
        "content": observation,
    })
    
    return conversation

def run_agent_loop(persona, conversation):

    MAX_SESSION_TOKENS = 60000
    MAX_AGENT_STEPS = 5
    parsed = None
    steps = 0
    session_tokens = 0

    while steps < MAX_AGENT_STEPS:
        steps += 1
        messages = build_messages_loop(conversation=conversation)

        session_tokens += llm.estimate_token(messages)

        if session_tokens > MAX_SESSION_TOKENS:
            raise RuntimeError (
                f"Session token budget exceeded:{session_tokens}"
            )

        # Call LLM
        parsed = single_step_decision(messages, persona)
        if parsed["type"] == "tool":
            conversation = execute_tool_from_parsed(parsed, conversation)
        else:
            return parsed
        
    if parsed is None:
        raise RuntimeError("Agent loop exited without producing a response")

    return parsed
