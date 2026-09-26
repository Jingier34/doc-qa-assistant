import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print("Loading model...")
start = time.time()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
print(f"Model loaded in {time.time() - start:.2f} seconds")

prompt = "Context: This is a test document for the RAG assistant project.\n\nQuestion: What is this document about?\n\nAnswer:"

print("Generating answer...")
start = time.time()
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Generation took {time.time() - start:.2f} seconds")
print("Answer:", answer)