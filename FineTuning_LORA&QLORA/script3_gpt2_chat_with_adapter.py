import os
import jsonlines
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

import warnings
warnings.filterwarnings('ignore')

BASE_MODEL = "gpt2"
ADAPTER_PATH = "models/adapters/gpt2_lora"
EVAL_FILE = "data/eval_questions.jsonl"


def load_eval_questions():
    questions = []
    with jsonlines.open(EVAL_FILE, "r") as reader:
        for obj in reader:
            questions.append(obj["question"])
    return questions


def clean_answer(text):
    if "Response:" in text:
        text = text.split("Response:", 1)[1].strip()
    for tok in ["\n", "(1)", "(2)", "(3)", "1.", "2.", "3."]:
        if tok in text:
            text = text.split(tok)[0].strip()
    return text.strip()


def chat(model, tokenizer, question):
    prompt = f"Instruction: {question}\nResponse:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return clean_answer(decoded)


def main():
    print("\n=== Loading GPT-2 base model and GPT-2 LoRA adapter... ===")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    print("\n=== Running GPT-2 adapter evaluation ===")
    questions = load_eval_questions()

    for q in questions:
        print(f"\nQuestion: {q}")
        ans = chat(model, tokenizer, q)
        print(f"Model Answer: {ans}")
        print("-" * 80)


if __name__ == "__main__":
    main()
