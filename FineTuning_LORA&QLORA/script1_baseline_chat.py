import jsonlines
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

import warnings
warnings.filterwarnings('ignore')

DATA_DIR = "data"
EVAL_FILE = os.path.join(DATA_DIR, "eval_questions.jsonl")

BASE_MODEL = "microsoft/phi-1.5"

def load_eval_questions():
    questions = []
    print("Reading eval file:", os.path.abspath(EVAL_FILE))
    with jsonlines.open(EVAL_FILE, "r") as reader:
        for obj in reader:
            questions.append(obj["question"])
    return questions

def chat(model, tokenizer, question, device):
    prompt = f"Questions: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt") #.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            temperature=0.2,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_token=True)
    return decoded


def run_baseline_evaluation(device):
    
    print(f"\nLoading base mdoel...(CPU may take some time)")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    ) #.to(device)
    questions = load_eval_questions()

    for q in questions:
        print(f"Question to the model -> {q}")
        answer = chat(model, tokenizer, q, device)
        print(f"Model's Answer: {answer}")
        print("-"*50)

if __name__=="__main__":
    device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    run_baseline_evaluation(device)