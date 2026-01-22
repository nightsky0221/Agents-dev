




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






TOOL_REGISTRY = {
    "calculator": calculator,
    # "search_docs": {
    #     "fn": search_docs,
    #     "schema": {
    #         "query": str
    #     }
    # },
    # "summarize_text": {
    #     "fn": summarize_text,
    #     "schema": {
    #         "text": str
    #     }
    # },
}







def validate_tool_request(tool_request:dict):
    if not isinstance(tool_request, dict):
        raise ValueError("Invalid tool_request format")
    
    tool_name = tool_request.get("tool")
    arguments = tool_request.get("arguments")

    if tool_name not in TOOL_REGISTRY:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a dict")
    
    return tool_name, arguments







def execute_tool(tool_name:str, arguments:dict):
    try:
        tool_fn = TOOL_REGISTRY[tool_name]
        result = tool_fn(**arguments)

        return {
            "status": "success",
            "tool": tool_name,
            "result": result,
        }
    
    except Exception as e:
        return{
            "status": "error",
            "tool": tool_name,
            "error": str(e),
        }