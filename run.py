import chatbot as c

# MEMORY TEST
# response = c.chat("My name is Alex and I work in fintech")
# response = c.chat("What do you remember about me?")
# response = c.chat("I love football and my major is NLP and LLM models.")
# response = c.chat("I have much experience with LLM-based projects and that is what I love now.")
# response = c.chat("So what do you rememeber about me now?")

# PLAN TEST
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")

# TOOL TEST
# response = c.chat("What is (5+7)?")

# PLANNING TEST
# response = c.chat("Give me a study plan to master LLM agents in 2 weeks.")

response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")
# response = c.chat("If I buy 3 books at $ 12 each and then add 10% tax, how much I do pay?")

if "answer" in response:
    print(response["answer"])
else:
    print("ERROR:", response)