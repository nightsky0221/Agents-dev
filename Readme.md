# AI Multi-Persona Chatbot & Agent Framework

This project is a **modular AI chatbot and agent framework** designed to demonstrate best practices for building **LLM-powered systems** with personas, guardrails, memory management, structured outputs, tool usage, and evaluation.

It is intended for **learning, experimentation, and prototyping** rather than production use.

---

## ✨ Key Features

- **Multi-persona routing**
  - Automatically routes user input to `tutor`, `support`, or `other` personas.
- **Agent loop with tool calling**
  - Supports safe tool execution (e.g. calculator).
- **Strict JSON schema enforcement**
  - Ensures predictable and machine-readable LLM outputs.
- **Conversation memory & summarization**
  - Compresses long conversations into summaries.
- **Confidence & quality evaluation**
  - Evaluates answers using a secondary LLM evaluator.
- **Prompt-injection guardrails**
  - Blocks common malicious prompt patterns.
- **Retry & validation logic**
  - Automatically retries invalid LLM responses.

---

## 🧠 Architecture Overview

```
User Input
   ↓
Guardrails (guardrails.py)
   ↓
Persona Router (router.py)
   ↓
Conversation Manager (chatbot.py)
   ↓
Agent Loop
   ├─ LLM Call (llm.py)
   ├─ JSON Validation (Json_structure.py)
   ├─ Tool Execution (tools.py)
   └─ Evaluation (check.py)
   ↓
Structured Response
```

---

## 📂 Project Structure

```
.
├── chatbot.py          # Main conversation + agent loop
├── router.py           # Persona routing logic
├── persona.py          # Persona system prompts
├── llm.py              # LLM interface & retry logic
├── Json_structure.py   # Output schema & JSON validation
├── tools.py            # Tool registry and execution
├── check.py            # Confidence checks & evaluation
├── summary.py          # Conversation summarization
├── guardrails.py       # Prompt-injection protection
├── run.py              # Example runner
└── README.md
```

---

## 🤖 Personas

| Persona  | Description |
|--------|-------------|
| Tutor  | Explains NLP & LLM concepts step-by-step |
| Support | Handles product and technical issues |
| Other  | Fallback for out-of-scope queries |

---

## 🔧 Tool System

Currently supported tools:

- **calculator**
  - Evaluates mathematical expressions safely
  - Example:
    ```json
    {
      "tool": "calculator",
      "arguments": { "expression": "23 * 17" }
    }
    ```

The agent loop automatically:
1. Detects tool requests
2. Validates arguments
3. Executes the tool
4. Feeds the result back to the LLM

---

## 📊 Evaluation & Confidence

Each response includes:
- `confidence` (0–1)
- `evaluation.score` (LLM-based quality judgment)
- `warnings` for:
  - Low confidence
  - Short or empty answers
  - Low evaluation scores

Decision logic determines whether to:
- Accept the response
- Warn the user
- Flag low confidence

---

## 🛡️ Safety & Guardrails

- Blocks common prompt-injection attempts
- Prevents memory poisoning
- Restricts tool execution
- Enforces strict JSON-only responses

---

## ▶️ How to Run

```bash
python run.py
```

Example:
```python
from chatbot import chat

response = chat("What is tokenization in NLP?")
print(response)
```

Structured JSON mode:
```python
response = chat_json("What is 23 * 17?")
```

---

## ⚠️ Known Limitations

- LLM backend is mocked (no real API connected)
- Some validation logic contains intentional bugs for practice
- Tool execution error handling is minimal
- Not production-hardened

---

## 🎯 Purpose

This project is ideal for:
- Learning **LLM system design**
- Practicing **agent architectures**
- Understanding **tool calling**
- Debugging **structured AI pipelines**
- Interview & portfolio demonstrations

---

## 📜 License

Educational / experimental use only.
