import router as rt
import persona as ps
import llm
import summary as suma

# create conversation store to read the user requests, send reponses and save historis by per persona.
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

# count the turns of conversation.
    turns = len(conversations[persona]) // 2

# after a certain turns, we start summarization to use limited memory efficiently
    if turns >= SUMMARY_TRIGGER:

        # extract system messages and cut the old messages from whole conversation histories.
        chunk = suma.extract_user_assistant_messages(conversations[persona][:-MAX_TURN])

        # summarize the old messages and limit the summarization up to its maximum size
        conversation_summaries[persona] = suma.clamp_summary(
            suma.update_summary(
                conversation_summaries[persona],
                chunk
            )
        )
        # keep the latest conversations
        conversations[persona] = conversations[persona][-MAX_TURN:]

    
    # Keep this priority order : persona > system > user, assistant

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

    # send the message to call the LLMs to get a response according to the request.
    response = llm.fake_llm(messages, persona)

    assistant_msg = {
        "role": "assistant",
        "persona": persona,
        "content": response
    }

    # add LLMs response to conversation
    conversations[persona].append(assistant_msg)
    global_conversation.append(assistant_msg)
    
    # limit the conversation amounts up to its maximum size
    conversations[persona] = trim_memory(conversations[persona])
    global_conversation = trim_global(global_conversation)

    # send the response to the user
    return response

# a reset function to initialize the conversation history
def reset_conversation():
    global global_conversation
    for persona in conversations:
        conversations[persona].clear()
    global_conversation.clear()