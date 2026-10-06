from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import chromadb
from fastapi.responses import FileResponse

app = FastAPI()

embedder = SentenceTransformer('all-MiniLM-L6-v2')

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
qa_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def serve_frontend():
    return FileResponse("index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()

    try:
        text = contents.decode("utf-8")
    except UnicodeDecodeError:
        return {"error": "File could not be read as text. Please upload a plain text (.txt) file."}

    if not text.strip():
        return {"error": "The uploaded file is empty."}

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
    if not question.strip():
        return {"error": "Question cannot be empty."}

    question_embedding = embedder.encode([question]).tolist()
    results = collection.query(query_embeddings=question_embedding, n_results=2)

    retrieved_chunks = results["documents"][0] if results["documents"] else []
    context = " ".join(retrieved_chunks)

    if not context:
        return {"answer": "No relevant content found. Please upload a document first."}

    prompt = f"Answer the question in a complete sentence based on the context below.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer in a full sentence:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = qa_model.generate(**inputs, max_new_tokens=100)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {
        "question": question,
        "retrieved_context": retrieved_chunks,
        "answer": answer
    }

@app.delete("/clear")
def clear_documents():
    global collection
    chroma_client.delete_collection(name="documents")
    collection = chroma_client.get_or_create_collection(name="documents")
    return {"message": "All documents cleared."}

@app.get("/documents")
def list_documents():
    all_data = collection.get()
    ids = all_data.get("ids", [])
    
    filenames = set()
    for doc_id in ids:
        filename = doc_id.rsplit("_", 1)[0]
        filenames.add(filename)
    
    return {"documents": sorted(filenames), "total_chunks": len(ids)}