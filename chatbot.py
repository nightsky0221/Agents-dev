import router as rt
import persona as ps
import llm
import summary as summarize
import Json_structure as js
from llm import call_llm_with_retries
import check as chk
import tools
import guardrails as gd

# create conversation store to read the user requests, send reponses and save histories by per persona.
conversations = {"tutor": [], "support": [], "other": []}
# create global conversation manager over personas
global_conversation = []

# limit the conversation history counts
def trim_global(messages, globe_max=50):
    return messages[-globe_max:]

def trim_memory(messages, max_memory=6):
    return messages[-max_memory:]

# define summarization variables
SUMMARY_TRIGGER = 8
MAX_TURN = 4

conversation_summaries = {
    "tutor": "",
    "support": "",
    "other": ""
}

# main chatbot function. decide the persona, manage communication between users and LLMs.
def chat(user_input, persona=None):

    global global_conversation, conversations, conversation_summaries

    try:
        gd.guard_input(user_input)
    except ValueError as e:
        return {"error": str(e)}

    if persona is None:
        persona = rt.route_persona(user_input)    

    if persona not in conversations:
        raise ValueError(f"Unknown persona: {persona}")
    
    user_msg = {
        "role": "user",
        "persona": persona,
        "content": user_input
    }

# add user input to conversation store for suitable persona.
    conversations[persona].append(user_msg)
    global_conversation.append(user_msg)

# after a certain turns, we start summarization to use limited memory efficiently
# *2 means that every conversation includes user and assistant dialogues for a couple
    if len(conversations[persona]) >= SUMMARY_TRIGGER * 2:

        # extract system messages and cut the old messages from whole conversation histories.
        old_messages = conversations[persona][:-MAX_TURN]
        chunk = summarize.extract_user_assistant_messages(old_messages)
        chunk_safe = [
            msg for msg in chunk
            if msg["role"] != "user" or summarize.is_memory_safe(msg["content"])
        ]

        # summarize the old messages and limit the summarization up to its maximum size
        conversation_summaries[persona] = summarize.clamp_summary(
            summarize.update_summary(
                conversation_summaries[persona],
                chunk_safe
            )
        )
        # keep the latest conversations
        conversations[persona] = conversations[persona][-MAX_TURN:]

    
    # Keep this priority order : persona(system) > summary(system) > user, assistant

    messages = [ps.personas[persona]]

    if conversation_summaries[persona]:
        messages.append({
            "role": "system",
            "content": f"Conversation summary:\n{conversation_summaries[persona]}"
        })
    
    messages.extend([
            {"role": m["role"], "content": m["content"]}
            for m in conversations[persona]
        ])
    
    # if "memory compression" in messages[0]["content"]:
    #     return "success"
    # return "normal"

    # tool request
    persona_prompt = ps.personas.get(persona, ps.personas["other"])
    summary_text = conversation_summaries[persona]

    # Run Agent loop
    parsed = run_agent_loop(
        persona=persona,
        persona_prompt=persona_prompt,
        summary=summary_text,
        conversation=conversations[persona],
        llm_call_fn=llm.llm_call
    )
    
    # ensure parsed exists
    if parsed is None:
        return {"error": "Agent loop returned no response"}

    # print("DEBUG parsed:", parsed)
    # using confidence check
    issues = chk.basic_confidence_check(parsed)
    if issues:
        parsed["warnings"] = issues

    # decision logic implementation
    action = chk.decide_response_action(parsed)

    if action =="warn":
        parsed["notice"] = (
            "This response may be unreliable. "
            "Consider asking a more specific question."
        )
  
    # refuse mode not now available, it's off by default now but later we will use that.
    # if should_refuse(parsed):
    #     parsed["answer"] = "I can't provide a reliable answer to that."
    #     parsed["confidence"] = 0.0


    # set the assistant messages
    assistant_msg = {
        "role": "assistant",
        "persona": persona,
        "content": parsed.get("answer", "")
    }

    # add LLMs response to conversation
    conversations[persona].append(assistant_msg)
    global_conversation.append(assistant_msg)
    
    # limit the conversation amounts up to its maximum size
    conversations[persona] = trim_memory(conversations[persona])
    global_conversation = trim_global(global_conversation)

    # send the response to the user
    return parsed

# a reset function to initialize the conversation history
def reset_conversation():
    global global_conversation
    for persona in conversations:
        conversations[persona].clear()
    global_conversation.clear()

# new chat function to use json_structure

def chat_json(user_input, persona=None):
    try:
        gd.guard_input(user_input)
    except ValueError as e:
        return {"error": str(e)}


    # Returns a JSON-structured response validated against OUTPUT_SCHEMA
    if persona is None:
        persona = rt.route_persona(user_input)

    messages = []

    persona_prompt = ps.personas.get(persona, ps.personas["other"])
    messages = [persona_prompt]
    messages.extend(js.build_json_prompt(user_input, persona_prompt))

    # Add retry failure
    raw_response = call_llm_with_retries(
        messages=messages,
        call_fn=llm.llm_call,
        persona=persona
    )
    parsed = js.parse_and_validate(raw_response)

    # using confidence check
    issues = chk.basic_confidence_check(parsed)
    if issues:
        parsed["warnings"] = issues

    # evaluation
    evaluation = chk.evaluate_answer(parsed["answer"], persona)
    
    if not isinstance(evaluation, dict):
        evaluation = {
            "score": 0.0,
            "issues": ["Evaluation missing or invalid"]
        }
    
    parsed["evaluation"] = evaluation or {
        "score": 0.0,
        "issues": ["Missing evaluation"]
    }
    parsed["action"] = chk.decide_response_action(parsed)

    return parsed

def build_messages(persona_prompt, summary, conversation):
    messages = [persona_prompt]

    if summary:
        messages.append({
            "role": "system",
            "content": f"Conversation summary:\n{summary}"
        })

    messages.extend([
        {"role": m["role"], "content": m["content"]}
        for m in conversation
    ])
    return messages

MAX_AGENT_STEPS = 3

def run_agent_loop(persona, persona_prompt, summary, conversation, llm_call_fn):
    parsed = None
    steps = 0

    while steps < MAX_AGENT_STEPS:
        messages = [
            persona_prompt,
            js.JSON_SYSTEM_PROMPT
        ]

        if summary:
            messages.append({
                "role": "system",
                "content": f"Conversation summary:\n{summary}"
            })
        
        messages.extend([
            {"role": m["role"], "content": m["content"]}
            for m in conversation
        ])

        raw = call_llm_with_retries(messages=messages, call_fn=llm_call_fn, persona=persona)
        parsed = js.parse_and_validate(raw)

        tool_request = parsed.get("tool_request")

        if not tool_request:
            break

        tool_name = tool_request.get("tool")
        arguments = tool_request.get("arguments", {})

        if tool_name not in tools.TOOLS:
            parsed["tool_error"] = f"Unknown tool: {tool_name}"
            break

        result = tools.TOOLS[tool_name]["function"](**arguments)

        if parsed is None:
            raise RuntimeError("Agent loop exited without producing a response")

        conversation.append({
            "role": "system",
            "content": f"Tool result ({tool_name}): {result}"
        })

        steps += 1

    return parsed