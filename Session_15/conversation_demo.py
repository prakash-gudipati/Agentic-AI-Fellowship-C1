"""
Demo 4 — Multi-turn conversation.

Wraps the LLMClient in a Conversation helper. The model sees the full
history on every turn and can reference earlier messages.

Run:
    $ python -m examples.conversation_demo
"""

from dotenv import load_dotenv
load_dotenv()

from llm_client import LLMClient, Conversation


client = LLMClient(provider="anthropic")
chat = Conversation(client, system="You are a concise tutor. Answer in one short paragraph.")

print("Turn 1 — user asks a definition question")
print("ASSISTANT:", chat.say("What is a vending machine?"))

print("\nTurn 2 — user asks a comparison; the model must remember turn 1")
print("ASSISTANT:", chat.say("How is it different from a fridge?"))

print("\nTurn 3 — user asks for a counter-example")
print("ASSISTANT:", chat.say("Give me one situation where a fridge would be the wrong choice."))

print("\n--- Full transcript ---")
print(chat.transcript())

print(f"\nTurns completed: {chat.turn_count()}")
print(f"Total cost: ${client.total_cost_usd:.6f}")
