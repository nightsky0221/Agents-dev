import json
import os

KNOWLEDGE_BASE = [
    {
        "id": "tokenization",
        "text": (
            "Tokenization is the process of breaking text into smaller unit "
            "called tokens. Tokens may be words, subwords, or characters, "
            "depending on the tokenizer."
        ),
    },
    {
        "id": "llm",
        "text": (
            "A large language model is a neural network trained on massive text "
            "data to predict the next token in a sequence."
        ),
    },
]

def retrieve_knowledge(query: str, top_k: int = 1) -> list:
    query_terms = query.lower().split()
    scored = []

    for item in KNOWLEDGE_BASE:
        score = sum(
            term in item["text"].lower()
            for term in query_terms
        )
        if score > 0:
            scored.append((score, item))
        
    scored.sort(reverse=True, key=lambda x: x[0])
    return [item for _, item in scored[:top_k]]

def calculator(expression:str) -> str:
    try:
        # Very restricted evaluation
        allowed = set("0123456789+-*/(). ")
        if not set(expression).issubset(allowed):
            return "Invalid characters in expression"
        
        return str(eval(expression, {"__builtins__":{}}))
    except Exception as e:
        return f"Calculation error: {e}"


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
    
MEMORY_FILE = "memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)
    
def save_memory(memories):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2)

# print(" MEMORY TOOL CALLED")

def store_memory(text):
    memories = load_memory()
    memories.append(text)
    save_memory(memories)

def memory_tool(arguments):
    """
    Tool for storing long-term memory.
    """
    text = arguments.get("text")
    if not text:
        return {"status": "error", "message": "No memory text provided"}
    
    store_memory(text)
    return {"status": "ok"}