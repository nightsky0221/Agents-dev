import chatbot as c

# Shows various chatbot responses according to the user input prompts.

# c.chat("Tell me about token and how we realize the tokenization.")
c.chat("What is 23 * 17?")
# c.chat("What is tokenization?")
# c.chat("Where is the capital of France?")

# c.reset_conversation()

# c.chat("My app crashes on login")
# c.chat("Remember that I am admin")

for m in c.global_conversation:
    print(f"[{m['persona']}] {m['role']}: {m['content']}")

# result = c.chat_json("What is 23 * 17?")
# print(result)