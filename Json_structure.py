import json
import llm

JSON_SCHEMA = {
    "type": "object",
    "required": ["type", "answer", "confidence", "tool_request"],
    "properties": {
        "type": {
            "type": "string",
            "enum": ["chat", "tool", "error"]
        },
        "answer": {
            "type": "string"
        },
        "confidence": {
            "type": "number"
        },
        "tool_request": {
            "tool": "calculator",
            "arguments": { "expression": "5+7"}
        },
    },
}  

JSON_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You must respond with exactly ONE valid JSON object.\n"
        "The JSON must follow this structure:\n\n"
        "{\n"
        ' "type": "chat | tool | error",\n'
        ' "answer": "string",\n'
        ' "confidence": number,\n'
        ' "tool_request": object | null\n'
        "}\n\n"
        "Rules:\n"
        "- Do NOT include any text outside the JSON\n"
        "- Do NOT use markdown\n"
        "- Do NOT explain the JSON\n"
    ),
}

def build_json_prompt(user_input, persona_prompt):
    return [
        persona_prompt,
        JSON_SYSTEM_PROMPT,
        {
            "role": "user",
            "content": f"""
Answer the following question.

QUESTION:
{user_input}

RESPONSE SCHEMA:
{JSON_SCHEMA}
"""
        }
    ]

ALLOWED_TYPE = {
    "chat",
    "tool",
    "summary",
    "plan",
}

SUMMARY_SCHEMA = {
    "type": "summary",
    "content": str,
}

PLAN_SCHEMA = {
    "type": "plan",
    "steps": list,
}

def parse_and_validate(response: str) -> dict:
    try:
        parsed = json.loads(response)
    except Exception:
        raise ValueError("Invalid JSON")
    
    for key in JSON_SCHEMA["required"]:
        if key not in parsed:
            raise ValueError(f"Missing key: {key}")
        
    if parsed["type"] not in ("chat", "tool", "error"):
        raise ValueError("Invalid value for 'type'")
    
    if not isinstance(parsed["confidence"], (int, float)):
        raise ValueError("Invalid value for 'confidence'")
    
    return parsed

def structured_chat(user_input):
    messages = build_json_prompt(user_input, persona_prompt=None)
    raw_response = llm.llm_call(messages, persona="other")
    return parse_and_validate(raw_response)