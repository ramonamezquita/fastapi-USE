from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_get_embeddings():
    params = {"sentence": "the quick brown fox"}
    response = client.get("/embeddings", params=params)
    assert response.status_code == 200
    assert "embedding" in response.json()


def test_get_embeddings_bulk():
    json = {
        "sentences": [
            "the quick brown fox jumped over the lazy dog",
            "the five boxing wizards jump quickly",
        ]
    }

    response = client.post("/embeddings/bulk", json=json)
    json = response.json()

    assert response.status_code == 200
    assert "embeddings" in json
    assert len(json["embeddings"]) == 2


def test_get_embeddings_similarity():
    json = {
        "sentence_1": "the quick brown fox jumped over the lazy dog",
        "sentence_2": "the five boxing wizards jump quickly",
    }

    response = client.post("/embeddings/similarity", json=json)
    json = response.json()

    assert response.status_code == 200
    assert "similarity" in json
    assert round(json["similarity"], 2) == 0.28
