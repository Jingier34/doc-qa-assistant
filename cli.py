import requests

BASE_URL = "http://127.0.0.1:8000"

def upload_document(filepath):
    with open(filepath, "rb") as f:
        response = requests.post(f"{BASE_URL}/upload", files={"file": f})
    return response.json()

def ask_question(question):
    response = requests.post(f"{BASE_URL}/query", params={"question": question})
    return response.json()

if __name__ == "__main__":
    print("=== Document Q&A Assistant ===")
    filepath = input("Enter path to a text file to upload: ").strip()

    print("\nUploading document...")
    upload_result = upload_document(filepath)
    print(f"Uploaded: {upload_result['filename']} ({upload_result['num_chunks']} chunks)\n")

    print("You can now ask questions about the document (type 'quit' to exit)\n")
    while True:
        question = input("Question: ").strip()
        if question.lower() == "quit":
            break
        result = ask_question(question)
        print(f"Answer: {result['answer']}\n")