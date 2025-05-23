import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load environment variables from .env file
load_dotenv()

# Fetch the Hugging Face token from env
hf_token = os.getenv("HF_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model_name = "google/gemma-3-1b-it"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)

input_conversation = [
    {"role": "user", "content": "the capital of France is"},
    {"role": "assistant", "content": "Paris"},
]

input_tokens=tokenizer.apply_chat_template(
    conversation=input_conversation,
    tokenizer=True,
)

model= AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

