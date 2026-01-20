
# 🧠 AI Chatbot & Agent Framework

A modular AI chatbot and agent framework showcasing **production-ready LLM system design**, including **persona-based reasoning, structured outputs, tool execution, memory management, and safety guardrails**.

This project demonstrates how to move from simple prompts to **reliable, extensible AI agents** suitable for real-world automation and decision support.

---

## 🔑 Highlights

- **Persona-Based Routing**  
  Automatically routes user queries to specialized personas (Tutor, Support, General).

- **JSON-Enforced LLM Responses**  
  Ensures structured, machine-readable outputs (`answer`, `confidence`, `tool_request`).

- **Tool-Calling Agent Loop**  
  Enables LLMs to request and execute tools (e.g., calculator) and reason over results.

- **Conversation Memory & Summarization**  
  Handles long conversations with automatic summarization and memory safety checks.

- **Safety & Guardrails**  
  Detects prompt injection and prevents unauthorized memory manipulation.

- **Confidence Scoring & Self-Evaluation**  
  Adds confidence estimates and automated response evaluation.

---

## 🏗️ High-Level Architecture

```
User Input
 → Guardrails
 → Persona Router
 → JSON-Enforced LLM
 → Agent Loop (Tool Execution)
 → Validation & Evaluation
 → Final Response
```

---

## ▶️ Example Usage

```python
import chatbot as c

result = c.chat("What is 23 * 17?")
print(result)
```

The system automatically detects the need for a tool, executes it, and returns a validated structured response.

---

## 🎯 Purpose

This project focuses on **AI system engineering**, emphasizing:
- Predictable outputs
- Agent-based reasoning
- Tool integration
- Safety-first design
- Extensible architecture

It reflects real-world patterns used in **LLM agents, automation systems, and enterprise AI platforms**.

---

## 👤 Author

**AI Engineer – NLP, LLMs, Agents & Automation**  
Specialized in building production-grade AI systems with structured outputs and autonomous workflows.
