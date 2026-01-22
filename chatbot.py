import router as rt
import persona as ps
import guardrails as gd
import agent_loop as aloop



# create conversation store to read the user requests, send reponses and save histories by per persona.
conversations = {"tutor": [], "support": [], "other": []}
# create global conversation manager over personas
global_conversation = []

# limit the conversation history counts
def trim_global(messages, globe_max=50):
    return messages[-globe_max:]

def trim_memory(messages, max_memory=6):
    return messages[-max_memory:]

MAX_TURNS_PER_PERSONA = 20

def chat(user_input, persona=None):

    global global_conversation, conversations

    try:
        gd.guard_input(user_input)
    except ValueError as e:
        return {"error": str(e)}

    if persona is None:
        persona = rt.route_persona(user_input)    

    if persona not in conversations:
        raise ValueError(f"Unknown persona: {persona}")
    
    if not conversations[persona]:
        conversations[persona].append(ps.personas[persona])
        
    user_msg = {
        "role": "user",
        "persona": persona,
        "content": user_input
    }

# add user input to conversation store for suitable persona.
    conversations[persona].append(user_msg)
    global_conversation.append(user_msg)

    # Run Agent loop

    if len(conversations[persona]) >= MAX_TURNS_PER_PERSONA:
        return {
            "answer": "This conversation has reached its limit. Please start a new session.",
            "confidence": 0.0,
            "tool_request": None,
            "error": {
                "code": "SESSION_LIMIT",
                "message": "Maximum conversation length reached",
                "retryable": False,
            }
        }

    try:
        parsed = aloop.run_agent_loop(
            persona=ps.personas[persona],
            conversation=conversations[persona],
        )
    except ValueError as e:
        parsed = {
            "type": "error",
            "answer": "I'm having trouble processing this right now. Please try again",
            "confidence": 0.0,
            "tool_request": None,
        }
    except Exception as e:
        print(f"[ERROR] {e}")
        parsed = {
            "type": "error",
            "answer": "Something went wrong, I couldn't complete your request safely.",
            "confidence": 0.0,
            "tool_request": None,
        }
    
    # ensure parsed exists
    if parsed is None:
        return {"error": "Agent loop returned no response"}

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

def chat_json(user_input, persona=None):
    try:
        gd.guard_input(user_input)
    except ValueError as e:
        return {"error": str(e)}


    # Returns a JSON-structured response validated against OUTPUT_SCHEMA
    if persona is None:
        persona = rt.route_persona(user_input)

    messages = []

    messages = aloop.build_messages_json(
        persona_prompt=ps.personas.get(persona, ps.personas["other"]),
        conversation=[{"role": "user", "content": user_input}]
    )

    # Add retry failure
    parsed = aloop.single_step_decision(messages, persona)

    return parsed