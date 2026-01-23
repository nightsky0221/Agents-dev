import llm
import json_structure as js
import tools
import guardrails as g
import summary as s

def single_step_decision(messages, persona):
    MAX_RETRIES = 3

    for _ in range(MAX_RETRIES):
        raw = llm.llm_call(messages, persona)

        try:
            return js.parse_and_validate(raw)
        except ValueError:
            messages.append({
                "role": "system",
                "content": (
                    "Your previous response was INVALID.\n"
                    "Respond again using ONLY the required JSON format.\n"
                    "Do not include any text outside the JSON."
                )
            })

    return {
        "type": "error",
        "answer": "LLM failed to produce valid JSON after retries",
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

    if conversation:
        user_query = conversation[-1]["content"]
        retrieved = tools.retrieve_knowledge(user_query)

        if retrieved:
            context_block = "\n\n".join(
                f"- {item['text']}" for item in retrieved
            )

            messages.append({
                "role": "system",
                "content": (
                    "Use the following context to answer the user's question. \n"
                    f"{context_block}"
                )
            })

    messages.extend([
        {"role": m["role"], "content": m["content"]}
        for m in conversation
    ])
    return messages

def execute_tool_from_parsed(parsed, conversation):
    tool_request = parsed.get("tool_request")

    if not tool_request:
        return conversation, None
    
    tool_name, arguments = tools.validate_tool_request(tool_request)
    observation = tools.execute_tool(tool_name, arguments)

    return conversation, {
        "type": "chat",
        "answer": str(observation["result"]),
        "confidence": 0.9,
        "tool_request": None
    }

def agent_loop(persona, conversation):

    MAX_SESSION_TOKENS = 60000
    MAX_AGENT_STEPS = 5
    parsed = None
    steps = 0
    session_tokens = 0
    plan = None

    if steps >= MAX_AGENT_STEPS:
        return {
            "type": "chat",
            "answer": "I was unable to complete the task within safe limits.",
            "confidence": 0.2,
            "tool_request": None
        }

    while steps < MAX_AGENT_STEPS:
        steps += 1

        if len(conversation) > 6:
            memory = s.summarize_conversation(conversation)
            conversation = [memory] + conversation[-4:]

        long_term = tools.load_memory()

        if long_term:
            conversation.insert(
                0,
                {
                    "role": "system",
                    "content": "Long-term memory:\n" + "\n".join(long_term[-5:])
                }
            )

        messages = build_messages_loop(conversation=conversation)

        session_tokens += llm.estimate_token(messages)

        if session_tokens > MAX_SESSION_TOKENS:
            raise RuntimeError (
                f"Session token budget exceeded:{session_tokens}"
            )

        # Call LLM
        parsed = single_step_decision(messages, persona)
        parsed = g.enforce_contract(parsed)

        if parsed["type"] == "summary":
            conversation.append({
                "role": "system",
                "content": parsed["content"]
            })
            continue

        if "plan" in parsed:
            conversation.append({
                "role": "assistant",
                "content": f"[PLAN] {parsed['plan']}"
            })

        if parsed["type"] == "plan":
            plan = parsed["steps"]
            
            conversation.append({
                "role": "system",
                "contemt": "Execution plan:\n" + "\n".join(
                    f"- {step}" for step in plan
                )
            })
            continue

        if parsed["type"] == "tool":
            conversation, final = execute_tool_from_parsed(parsed, conversation)

            if final:
                return final
            continue

        parsed = g.enforce_contract(parsed)
        return parsed

        
    if parsed is None:
        raise RuntimeError("Agent loop exited without producing a response")
    
    parsed = g.enforce_contract(parsed)
    return parsed
