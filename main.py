from fastapi import FastAPI, UploadFile, File
from sentence_transformers import SentenceTransformer
import chromadb

app = FastAPI()

# 加载embedding模型（只在启动时加载一次）
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# 初始化向量数据库
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="documents")

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    text = contents.decode("utf-8")

    # 简单切块：每200个字符一块
    chunk_size = 200
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # 生成embedding并存入向量库
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return {
        "filename": file.filename,
        "text_length": len(text),
        "num_chunks": len(chunks),
        "preview": text[:200]
    }