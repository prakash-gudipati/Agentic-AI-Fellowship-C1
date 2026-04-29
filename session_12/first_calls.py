import os
from dotenv import load_dotenv

import anthropic
import openai

load_dotenv()

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is not set")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

PRICING = {
    "claude-opus-4-7":     {"input": 5.00,  "output": 25.00},
    "claude-sonnet-4-6":   {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5":    {"input": 1.00,  "output": 5.00},
    "gpt-5":               {"input": 1.25,  "output": 10.00},
    "gpt-5-mini":          {"input": 0.25,  "output": 2.00},
    "gpt-4.1":             {"input": 2.00,  "output": 8.00},
    "gpt-4.1-nano":        {"input": 0.10,  "output": 0.40},
}

def estimate_cost(input_tokens: int, output_tokens: int, model: str) -> float:
    if model not in PRICING:
        return 0.0
    rates = PRICING[model]
    input_cost = (input_tokens / 1_000_000) * rates["input"]
    output_cost = (output_tokens / 1_000_000) * rates["input"]
    return input_cost + output_cost

def ask_claude(prompt: str,
               model: str,
               temperature: float = 0.7,
               max_tokens: int = 300,
               system: str = "You are a helpful assistance. Send one prompt Claude and return the reply + metadata"):
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    try:
        response = client.messages.create(
            model = model,
            max_tokens= max_tokens,
            temperature=temperature,
            system=system,
            messages= [{"role": "user", "content": prompt}]
        )
    except anthropic.AuthenticationError:
        raise RuntimeError("Claude has rejected your API key")
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"Claude API connection failed with error : {e}")
    
    text = response.content[0].text
    return {
        "provider": "Anthropic",
        "model": response.model,
        "reply": text,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "stop_reason": response.stop_reason
    }

def ask_openai(prompt: str,
               model: str,
               temperature: float = 0.7,
               max_tokens: int = 300,
               system: str = "You are a helpful assistance. Send one prompt Claude and return the reply + metadata"):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model = model,
            temperature= temperature,
            max_tokens=max_tokens,
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
        )
    except openai.AuthenticationError:
        raise RuntimeError("Please check your Open AI API key")
    except openai.APIConnectionError as e:
        raise RuntimeError(f"Unable to reach OpenAI API: {e}")
    text = response.choices[0].message.content
    return{
        "provider": "OpenAI",
        "model": response.model,
        "reply": text,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "stop_reason": response.choices[0].finish_reason
    }
    
def main():
    prompt = "In two sentences, explain about Large Language Models"

    print("=" * 50)
    print("First API CALLS TO LLMs")
    print("=" * 50)

    claude_result = ask_claude(prompt, model="claude-sonnet-4-6", temperature=0.7)
    print(claude_result)

    openai_result = ask_openai(prompt, model="gpt-4.1", temperature=0.7)
    print(openai_result)

    total = estimate_cost(claude_result["input_tokens"],claude_result["output_tokens"],"claude-sonnet-4-6") 
    + estimate_cost(openai_result["input_tokens"],openai_result["output_tokens"],"gpt-4.1")

    print(f"\n Session total cost for this runs : ${total:.6f}")    

if __name__ == "__main__":
    main()
