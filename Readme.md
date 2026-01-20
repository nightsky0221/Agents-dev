# Chatbot Architecture and Persona-Driven LLM Management

## Overview

This project presents a modular, persona-driven chatbot architecture designed to build **controlled, scalable, and maintainable conversational systems** on top of Large Language Models (LLMs).

Rather than focusing on basic LLM invocation, the system addresses the core production challenge: **constraining and stabilizing LLM behavior over long-running conversations**. The architecture prioritizes predictability, persona fidelity, security, and operational robustness, making it suitable as a reference implementation for real-world deployments.

---

## System Architecture

The chatbot is implemented as a **multi-stage processing pipeline** with clearly separated responsibilities:

1. User input ingestion and intent analysis
2. Persona selection and routing
3. Prompt construction with behavioral constraints
4. Persona-conditioned LLM response generation
5. Conversation memory management and summarization
6. Response delivery and history inspection

This separation of concerns enables extensibility, testability, and long-term maintainability.

---

## Persona Routing and Prompt Control

Incoming user input is analyzed and routed to a fixed set of personas:

- **Tutor** — Educational queries related to NLP and LLMs
- **Support** — Product usage and troubleshooting requests
- **Other** — Out-of-scope or ambiguous inputs

Routing is implemented via a lightweight keyword-based mechanism, which can be replaced by a more advanced intent classifier without affecting downstream logic.

Once a persona is selected, the system constructs a deterministic prompt consisting of:
- A persona-specific system prompt (injected on every call)
- A short-term buffer of recent conversation turns
- An optional long-term conversation summary

This ensures consistent persona behavior while controlling token usage and preventing context overflow.

---

## Persona-Conditioned Response Generation

LLM responses are generated based on:
- The active persona
- The current user request
- Retained conversational context (recent turns plus summary memory)

This enables role-appropriate behavior, such as structured explanations for the Tutor persona and concise, action-oriented responses for the Support persona. Persona logic is fully decoupled from routing and memory management, allowing new personas to be added with minimal changes.

---

## Memory Management and Summarization

To support long-running conversations, the system implements a **two-tier memory model**:

- **Short-term memory** — Recent turns stored verbatim
- **Long-term memory** — Older history compressed into summaries

When a configurable threshold is exceeded, older messages are summarized to preserve key facts, goals, preferences, and constraints while significantly reducing token usage.

Summarization is orchestrated with safeguards to prevent:
- Hallucinated or distorted memory
- Persona override
- Unbounded summary growth
- Stale or conflicting user intent

---

## Security, Reliability, and Policy Enforcement

The system enforces strict behavioral guarantees through dedicated validation and control layers implemented across modular Python components (e.g., `chatbot.py`, `llm.py`).

### Key guarantees include:

**Structural Integrity**
- Strict JSON output contracts
- Schema validation
- No silent corrections

**Security**
- Prompt injection detection and blocking
- Memory poisoning prevention
- Persona lock-in enforcement

**Reliability**
- Retry mechanisms and controlled fallbacks
- Explicit failure modes
- Confidence checks and self-evaluation logic

Decision logic determines whether responses are delivered, retried, warned, or refused based on policy compliance.

---

## Conclusion

This project provides a production-oriented reference for building **persona-driven, policy-controlled LLM systems** with robust context management and long-term conversational stability.

It demonstrates how LLMs can be managed predictably and responsibly in extended, multi-persona interactions and serves as a strong foundation for more advanced conversational agents.

If you encounter issues or edge cases, please report them—continuous iteration and improvement are core principles of this project.