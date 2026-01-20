def calculator(expression:str) -> str:
    try:
        return str(eval(expression, {"__builtins__": {}}))
    except Exception:
        return "Calculation error"

TOOLS = {
    "calculator": {
        "description": "Evaluate a math expression",
        "function": calculator,
        "input_schema": {
            "expression": "string"
        }
    }
}

def validate_tool_request(tool_request:dict) -> dict:
    if not isinstance(tool_request, dict):
        raise ValueError("Invalid tool_request format")
    
    tool_name = tool_request.get("tool")
    arguments = tool_request.get("arguments")

    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")
    
    return tool_name, arguments

def execute_tool(tool_name:str, arguments:dict) -> str:
    tool_fn = TOOLS[tool_name]["function"]
    return tool_fn(**arguments)