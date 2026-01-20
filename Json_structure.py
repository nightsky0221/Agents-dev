import json
import llm

OUTPUT_SCHEMA = {
    "answer": "string",
    "confidence": "number between 0 and 1",
    "tool_request": {
        "tool": "string",
        "arguments": "object"
    },
}

JSON_SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You must respond ONLY in valid JSON. "
        "Do not include explanations, comments, or extra text. "
        "The JSON must strictly follow the provided schema. "
        "- calculator(expression: string)\n"
        "If no tool is needed, set tool_request to null. "
    )
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
{OUTPUT_SCHEMA}
"""
        }
    ]

def parse_and_validate(response):
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON")
    
    if "answer" not in data or "confidence" not in data:
        raise ValueError("Schema violation")
    
    if not isinstance(data["confidence"], (int, float)):
        raise ValueError("Confidence must be numeric")
    
    if not (0 <= data["confidence"] <= 1):
        raise ValueError("Confidence our of range")
    
    return data

def structured_chat(user_input):
    messages = build_json_prompt(user_input, persona_prompt=None)
    raw_response = llm.llm_call(messages, persona="other")
    return parse_and_validate(raw_response)

TOOL_REQUEST_SCHEMA = {
    "tool": "string",
    "arguments": "object"
}