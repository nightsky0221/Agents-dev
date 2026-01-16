import chatbot

# Shows various chatbot responses according to the user input prompts.

chatbot.chat("Tell me about token and how we realize the tokenization.")
chatbot.chat("Where is the capital of France?")

# for m in conversations["tutor"]:
#     print(m["role"], ": ", m["content"])

# print("\n---\n")

# for m in conversations["other"]:
#     print(m["role"], ": ", m["content"])

# print("\n---\n")

# reset_conversation()

chatbot.chat("My app crashes on login")

# for m in conversations["support"]:
#     print(m["role"], ": ", m["content"])

# print("\nTutor persona instructions:\n")
# print(personas["tutor"]["content"])

for m in chatbot.global_conversation:
    print(f"[{m['persona']}] {m['role']}: {m['content']}")