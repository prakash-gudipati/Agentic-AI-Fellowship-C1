import json
import os
import tiktoken
from pathlib import Path

from utils.text_processor import(
    clean_text,
    split_paragraphs,
    split_sentences
)

ENCODER_NAME = "cl100k_base"
HAIKU_RATE_PER_M = 1.00
GPT_4O_RATE_PER_M = 2.50
O1_RATE_PER_M = 15.00

DEFAULT_INPUT_PATH = Path("data/messy_article.txt")
DEFAULT_OUTPUT_PATH = Path("data/cleaned_output.json")

def count_tokens(text: str, encodername:str = ENCODER_NAME) -> int:
    encoder = tiktoken.get_encoding(encodername)
    token_ids = encoder.encode(text)
    return len(token_ids)

def estimate_cost(token_count:int) -> dict[str,float]:
    return {
        "haiku" : round(token_count * HAIKU_RATE_PER_M / 1_000_000 , 6),
        "Gpt-40" : round(token_count * GPT_4O_RATE_PER_M / 1_000_000 , 6),
        "o1" : round(token_count * O1_RATE_PER_M / 1_000_000 , 6)
    }

def read_text_file(path: Path) -> str:
    print(path)
    if not path.exists():
        return FileNotFoundError("Please check the path")
    with open(path, "r", encoding = "utf-8") as f:
        return f.read()

def write_json(path:Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def process_document(input_path:Path, output_path:Path) -> dict:
    raw = read_text_file(input_path)
    cleaned = clean_text(raw)
    paragraphs = split_paragraphs(cleaned)
    sentences = split_sentences(cleaned)

    char_count = len(cleaned)
    word_count = len(cleaned.split())
    token_count = count_tokens(cleaned)
    cost = estimate_cost(token_count)

    report = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "original_chars": len(raw),
            "cleaned_chars": len(cleaned),
            "word_count": word_count,
            "chars_count": char_count,
            "sentence_count": sentences,
            "cleaned_text": cleaned,
            "paragraphs": paragraphs,
            "estimated_cost": cost
        }
    
    write_json(output_path, report)


def main():
    input_path = DEFAULT_INPUT_PATH
    output_path = DEFAULT_OUTPUT_PATH

    print(f"Processing the text pipline with encoder: {ENCODER_NAME}")

    process_document(input_path,output_path)

    print("Processing Done!!")


if __name__ == "__main__":
    main()