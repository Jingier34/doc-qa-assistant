from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb

app = FastAPI()

embedder = SentenceTransformer('all-MiniLM-L6-v2')

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")

    chunk_size = 200
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    return {
        "filename": file.filename,
        "text_length": len(text),
        "num_chunks": len(chunks),
        "preview": text[:200]
    }

@app.post("/query")
async def query(question: str):
    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=2)

    retrieved_chunks = results["documents"][0] if results["documents"] else []
    context = " ".join(retrieved_chunks)

    if not context:
        return {"answer": "No relevant content found. Please upload a document first."}

    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = qa_model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "question": question,
        "retrieved_context": retrieved_chunks,
        "answer": answer
    }