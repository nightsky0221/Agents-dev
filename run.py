import chatbot as c

response = c.chat("Tell me about the tokenization.")

if "answer" in response:
    print(response["answer"])
else:
    print("ERROR:", response)