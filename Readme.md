# Portfolio-Grade Autonomous AI Agent

A fully self-hosted, **framework-free autonomous AI agent** built on top of an open-source LLM (LLaMA via Ollama).  
This project demonstrates **production-grade agent architecture** including strict JSON contracts, tool execution, short‑term & long‑term memory, planning, safety guardrails, and bounded execution.

This system was intentionally built **without LangChain or external agent frameworks** to demonstrate deep understanding of how agents work internally.

---

## 🚀 Key Features

- **Strict JSON Contract Enforcement**  
  All LLM outputs are validated against a schema before execution.

- **Tool-Using Agent**  
  The agent can invoke tools (e.g. calculator, memory) via structured JSON.

- **Short-Term Memory (Session)**  
  Conversation summarization prevents context overflow.

- **Long-Term Memory (Persistent)**  
  Important user facts are stored in `memory.json` and persist across restarts.

- **Autonomous Planning**  
  The agent can generate internal plans and execute them step-by-step.

- **Safety Guardrails**  
  Prompt-injection detection, bounded loops, token limits, and graceful failure handling.

- **Self-Hosted LLM**  
  Uses Ollama + LLaMA locally (no paid APIs).

---

## 🧠 Architecture Overview

```
User Input
   ↓
Persona Routing (router.py)
   ↓
Agent Loop (agent_loop.py)
   ├─ Memory Injection (summary + memory.json)
   ├─ Planning (type=plan)
   ├─ Tool Execution (type=tool)
   ├─ Safety Enforcement (guardrails)
   ↓
Validated JSON Response
```

The **agent loop is the brain** of the system. Every decision flows through it.

---

## 📁 Project Structure

```
.
├── agent_loop.py        # Core agent execution loop
├── chatbot.py           # Conversation manager
├── guardrails.py        # Input validation & contract enforcement
├── json_structure.py    # JSON schema & validation logic
├── llm.py               # Ollama / LLaMA interface + retry logic
├── persona.py           # System personas & behavioral rules
├── router.py            # Persona routing logic
├── summary.py           # Short-term memory summarization
├── tools.py             # Tool registry (calculator, memory)
├── memory.json          # Persistent long-term memory (auto-created)
├── run.py               # Entry point / test harness
└── README.md
```

---

## ⚙️ Requirements

- Python **3.10+**
- Ollama installed locally
- LLaMA model pulled (example):

```bash
ollama pull llama3
```

---

## ▶️ How to Run

1. **Start Ollama**
```bash
ollama serve
```

2. **Run the agent**
```bash
python run.py
```

3. Modify `run.py` to test:
- memory persistence
- tool usage
- planning behavior

---

## 🧪 Example Capabilities

### Tool Use (Calculator)
```
User: What is (5 + 7)?
Agent → tool: calculator
```

### Long-Term Memory
```
User: I work in fintech.
(restart program)
User: What do you remember about me?
```

### Planning
```
User: Give me a 2-week plan to master LLM agents.
Agent → type=plan → executes → final answer
```

---

## 🛡️ Safety & Reliability

- **Prompt injection detection**
- **Strict schema validation**
- **Retry + fallback on LLM failure**
- **Max step & token limits**
- **Guaranteed termination**

This ensures the agent never crashes or loops indefinitely.

---

## 🎯 Design Philosophy

- Explicit over implicit
- Contracts over trust
- Control flow > prompt magic
- Debuggable > fancy

Every component is transparent, inspectable, and replaceable.

---

## 📌 Portfolio Value

This project demonstrates:

- Deep understanding of LLM control flow
- Real agent architecture (not demos)
- Tool calling & memory done correctly
- Production-minded failure handling

This is suitable for:
- AI Engineer roles
- LLM / Agent research
- Systems design interviews

---

## 🔮 Possible Extensions

- Vector-based memory retrieval
- Multi-agent coordination
- Evaluation harness
- Logging & tracing
- Web or API interface

---

## 🧑‍💻 Author

Built as part of an advanced AI engineering curriculum focused on:
**NLP · LLMs · Autonomous Agents · Automation**

---

## 📄 License

MIT (or specify your own)

