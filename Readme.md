
# Modular AI Chatbot & Agent Framework

## Overview

This project is a **modular AI chatbot and agent framework** built on top of a local Large Language Model (LLM) using **Ollama**.  
It demonstrates how to design a structured, safe, and extensible AI system with persona routing, tool usage, and controlled agent loops.

The framework is intended for **learning, experimentation, and prototyping** modern LLM-based systems rather than direct production deployment.

---

## Core Capabilities

- **Persona-Based Conversation Handling**
  - Automatically routes user queries to the most appropriate persona.
- **Agent Loop Architecture**
  - Enables multi-step reasoning with bounded execution.
- **Tool Invocation**
  - Allows the LLM to request and execute tools through a validated interface.
- **Strict JSON Output Enforcement**
  - Ensures predictable and machine-readable responses.
- **Prompt Injection Guardrails**
  - Detects and blocks common prompt-injection attempts.
- **Conversation Memory Management**
  - Maintains scoped memory per persona with size limits.
- **Robust Error Handling & Retries**
  - Handles malformed model outputs safely.

---

## Project Structure

```
.
├── agent_loop.py        # Core agent loop logic
├── chatbot.py           # Chat interface and conversation manager
├── guardrails.py        # Prompt injection protection
├── Json_structure.py    # JSON schema definitions and validation
├── llm.py               # LLM wrapper (Ollama integration)
├── persona.py           # Persona definitions and system prompts
├── router.py            # Persona routing logic
├── tools.py             # Tool registry and execution
├── run.py               # Example entry point
└── README.md
```

---

## System Architecture

```
User Input
   ↓
Guardrails (Injection Detection)
   ↓
Persona Router
   ↓
Conversation Store
   ↓
Agent Loop
   ├─ LLM Call
   ├─ Tool Request (Optional)
   └─ Tool Execution
   ↓
Validated JSON Response
```

---

## Personas

Defined in `persona.py`:

- **Tutor**
  - Explains NLP, LLM, and AI concepts step-by-step.
- **Support**
  - Handles error reports, bugs, and product-related issues.
- **Other**
  - Default fallback persona.

Each persona injects a system-level instruction to guide LLM behavior.

---

## Tools

Tools are defined in `tools.py` and executed only after validation.

### Available Tools

- **calculator**
  - Safely evaluates mathematical expressions.

Tools are:
- Explicitly requested by the LLM
- Validated against a registry
- Executed in a controlled environment

---

## JSON Response Schema

All structured responses must follow this schema:

```json
{
  "type": "chat | tool | error",
  "answer": "string",
  "confidence": 0.0,
  "tool_request": null | {
    "tool": "string",
    "arguments": {}
  }
}
```

This ensures:
- Predictable outputs
- Safe downstream consumption
- Reliable tool invocation

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Ollama installed and running
- `llama3` model pulled locally

```bash
ollama pull llama3
```

---

### Running the Example

```bash
python run.py
```

Example usage:

```python
response = chat("Explain tokenization in NLP.")
print(response["answer"])
```

---

## Intended Use Cases

- Learning how LLM-based chat systems are structured
- Experimenting with agent loops and tool calling
- Building structured JSON-based AI APIs
- Prototyping persona-driven chatbots

---

## Limitations

- Token estimation is approximate
- Tools are synchronous
- JSON compliance depends on LLM output quality
- Not optimized for production workloads

---

## Future Enhancements

- Asynchronous tool execution
- Additional tools (search, summarization, RAG)
- Persistent storage for conversation memory
- Streaming responses
- Improved token accounting

---

## License

This project is provided for **educational and experimental purposes**.
