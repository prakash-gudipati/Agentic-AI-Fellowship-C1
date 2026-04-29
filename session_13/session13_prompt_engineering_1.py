import os
from dotenv import load_dotenv
import json
import re
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 400
LOW_TEMP = 0.2
HIGH_TEMP = 0.9
PRODUCT_NAME = "AUDIWARE X1 WIRELESS HEADPHONES"
PRODUCT_FACTS = {
    "40-hour battery life; noise cancelling; 250g weight; memory-foam ear cushions; built for daily commuters"
}
TARGET_WORD_LOW = 50
TARGET_WORD_HIGH = 70

def make_client():
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def call_claude(client,system_prompt,user_prompt,temperature=LOW_TEMP):
    try:
        response = client.messages.create(
            model = MODEL,
            max_tokens = MAX_TOKENS,
            temperature = temperature,
            system = system_prompt,
            messages = [{"role": "user","content": user_prompt}]
        )
        return response.content[0].text.strip()
    except Exception as error:
        return f"[Error Communication with Claude]: {type(error).__name__}: {error}"

def prompt_v1_zero_shot():
    system = "You are a helpful assistant"
    user = f"Write product description for {PRODUCT_NAME}"
    return system, user, LOW_TEMP

def prompt_v2_few_shot():
    system = "You are a helpful assistant"
    user = (
        "Here are three product descriptions in the style we want.\n\n"
        "Example 1 — Smartwatch:\n"
        "Track every heartbeat, every step, every climb. The Pulse Pro slips "
        "on like a feather and survives whatever your week throws at it. "
        "Three-day battery. Five colours. One reason to leave the desk.\n\n"
        "Example 2 — Backpack:\n"
        "Built for the 7am train and the 9pm flight. The Voyager 22 carries a "
        "laptop, a change of clothes, and your charger without weighing you "
        "down. Water-resistant. Lifetime guarantee.\n\n"
        "Example 3 — Coffee maker:\n"
        "Wake up to coffee that smells like a cafe. The Brew One grinds the "
        "beans, heats the water, and times the pour while you find your "
        "shoes. Twelve cups. One button.\n\n"
        f"Now write one for: {PRODUCT_NAME}.\n"
        f"Facts to use: {PRODUCT_FACTS}"
    )
    return system,user,LOW_TEMP

def prompt_v3_role():
    system = ("You are a sesion copywriter ata a premium audio brand. "
              "Your descriptions are warm, confident and concrete - never generic.")
    user = (
        f"Write the product description for : {PRODUCT_NAME}"
        F"Facts to use: {PRODUCT_FACTS}"
    )
    return system,user, LOW_TEMP

def prompt_v4_delimiters():
    system = ("You are a sesion copywriter ata a premium audio brand. "
              "Your descriptions are warm, confident and concrete - never generic.") 
    user = (
        "### INSTRUCTIONS\n"
        "Write a product description.\n\n"
        "### CONTEXT\n"
        "Audience: daily commuters who already own decent earbuds and want "
        "an upgrade.\n\n"
        "### INPUT\n"
        f"Product: {PRODUCT_NAME}\n"
        f"Facts: {PRODUCT_FACTS}\n\n"
        "### LENGTH\n"
        f"Between {TARGET_WORD_LOW} and {TARGET_WORD_HIGH} words."
    )
    return system, user, LOW_TEMP

def prompt_v5_output_format():
    system = ("You are a sesion copywriter ata a premium audio brand. "
              "Your descriptions are warm, confident and concrete - never generic."
              "You always reply with valid JSON and nothing else")
    user = (
        "### INSTRUCTIONS\n"
        "Write a product description and return it as JSON.\n\n"
        "### CONTEXT\n"
        "Audience: daily commuters upgrading from cheap earbuds.\n\n"
        "### INPUT\n"
        f"Product: {PRODUCT_NAME}\n"
        f"Facts: {PRODUCT_FACTS}\n\n"
        "### OUTPUT FORMAT\n"
        "{\n"
        '  "title": "<8 words or fewer>",\n'
        f'  "body":  "<{TARGET_WORD_LOW}-{TARGET_WORD_HIGH} words>",\n'
        '  "tags":  ["<tag1>", "<tag2>", "<tag3>"]\n'
        "}\n"
        "Reply with the JSON only. No prose before or after."
    )
    return system,user,LOW_TEMP

def prompt_v6_chain_of_thought():
    system = ("You are a sesion copywriter ata a premium audio brand. "
              "Your descriptions are warm, confident and concrete - never generic."
              "You always reply with valid JSON and nothing else")
    user = (
        "### INSTRUCTIONS\n"
        "Before writing, think step by step inside the JSON field 'thinking'. "
        "Then write the final description.\n\n"
        "### CONTEXT\n"
        "Audience: daily commuters upgrading from cheap earbuds.\n\n"
        "### INPUT\n"
        f"Product: {PRODUCT_NAME}\n"
        f"Facts: {PRODUCT_FACTS}\n\n"
        "### OUTPUT FORMAT\n"
        "{\n"
        '  "thinking": "<step 1: who is the audience? '
        'step 2: which fact matters most? step 3: what feeling do we want?>",\n'
        '  "title":    "<8 words or fewer>",\n'
        f'  "body":     "<{TARGET_WORD_LOW}-{TARGET_WORD_HIGH} words>",\n'
        '  "tags":     ["<tag1>", "<tag2>", "<tag3>"]\n'
        "}\n"
        "Reply with the JSON only."
    )    
    return system,user,LOW_TEMP

if __name__ == "__main__":
    client = make_client()
    system,user,temp = prompt_v6_chain_of_thought()
    print(call_claude(client,system,user,temp))