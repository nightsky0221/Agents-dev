import json
import llm






OUTPUT_SCHEMA = {
    "answer": "string",
    "confidence": "number between 0 and 1",
    "tool_request": {
        "tool": "string",
        "arguments": "object"
    },
    "warnings": {
        "type": "array",
        "items": {"type": "string"},
        "default": []
    },
    "evaluation": {
        "type": "object",
        "properties": {
            "score": {"type": "number"},
            "issues": {
                "type": "array",
                "items": {"type": "string"}
            },
        },
        "required": ["score", "issues"]
    },
    "action": {
        "type": "string"
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






def parse_and_validate(response) -> dict:
    if isinstance(response, dict):
        validate_schema(response)
        return response
    
    if not isinstance(response, str):
        raise ValueError("LLM output must be str or dict")
    
    try:
        data = json.loads(response)
    except json.JSONDecodeError as e:
        raise ValueError("Invalid JSON from LLM") from e
    
    validate_schema(data)
    
    if "answer" not in data or "confidence" not in data:
        raise ValueError("Schema violation")
    
    if not isinstance(data["confidence"], (int, float)):
        raise ValueError("Confidence must be numeric")
    
    if not (0 <= data["confidence"] <= 1):
        raise ValueError("Confidence our of range")
    
    return data







def validate_schema(parsed: dict):
    if not isinstance(parsed, dict):
        raise ValueError("Parsed output must be a dict")
    
    if "type" not in parsed:
        raise ValueError("Missing required field: type")
    
    if parsed[type] not in ("chat", "tool", "error"):
        raise ValueError(f"Invalid type: {parsed['type']}")
    
    if "content" not in parsed:
        raise ValueError("Missing required field: content")
    
    # tool responses must declare tool_name and arguments
    if parsed["type"] == "tool":
        if not parsed.get("tool_name"):
            raise ValueError("Tool response missing tool_name")
        if "arguments" not in parsed:
            raise ValueError("Tool response missing arguments")
    
    # optional but normalized
    if "warnings" not in parsed:
        parsed["warnings"] = []

    return parsed









def structured_chat(user_input):
    messages = build_json_prompt(user_input, persona_prompt=None)
    raw_response = llm.llm_call(messages, persona="other")
    return parse_and_validate(raw_response)







TOOL_REQUEST_SCHEMA = {
    "tool": "string",
    "arguments": "object"
}