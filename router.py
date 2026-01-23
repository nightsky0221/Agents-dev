from guardrails import guard_input
import tools

def route_persona(user_input):

    user_input = guard_input(user_input)
    text = user_input.lower().replace("'", "")

    support_keywords = [
        "crash", "error", "bug", "login", "issue", "problem", "doesnt work", "failed", "help", "support"
    ]
    tutor_keywords = [
        "explain", "what is", "how does", "token", "embedding", "llm", "nlp", "transformer", "attention", "model"
    ]
    
    if any(k in text for k in support_keywords):
        return "support"
    elif any(k in text for k in tutor_keywords):
        return "tutor"
    else:
        return "other"

def route(decision):
    if decision["type"] == "chat":
        return decision
    
    if decision["type"] == "tool":
        try:
            tool_name, arguments = tools.validate_tool_request(decision["tool_request"])
            observation = tools.execute_tool(tool_name, arguments)

            return {
                "type": "chat",
                "answer": str(observation["result"]),
                "confidence": 0.9,
                "tool_request": None
            }
        
        except Exception as e:
            return {
                "type": "error",
                "answer": str(e),
                "confidence": 0.0,
                "tool_request": None
            }
           
    return decision