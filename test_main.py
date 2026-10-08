from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_upload_empty_file():
    response = client.post(
        "/upload",
        files={"file": ("empty.txt", b"", "text/plain")}
    )
    assert response.status_code == 200
    assert response.json()["error"] == "The uploaded file is empty."

def test_upload_valid_file():
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"This is a test document.", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["num_chunks"] >= 1

def test_query_empty_question():
    response = client.post("/query", params={"question": ""})
    assert response.status_code == 200
    assert response.json()["error"] == "Question cannot be empty."

def test_clear_documents():
    response = client.delete("/clear")
    assert response.status_code == 200
    assert "cleared" in response.json()["message"].lower()