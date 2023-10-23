import os

from fastapi import FastAPI
from numpy import dot
from numpy.linalg import norm

from . import modelclient, models


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    return dot(v1, v2) / (norm(v1) * norm(v2))


app = FastAPI()


def create_model_client() -> modelclient.TensorFlowServingClient:
    """Creates tensorflow serving model client."""
    model_endpoint = os.getenv("MODEL_ENDPOINT")
    if model_endpoint is None:
        raise ValueError("`MODEL_ENDPOINT` environment variable not set.")

    return modelclient.TensorFlowServingClient(model_endpoint)


model_client = create_model_client()


@app.get("/embeddings/", response_model=models.Embedding)
async def get_embeddings(sentence: str):
    """Returns a JSON encoded array of the embedding for the given sentence."""
    embedding = model_client.predict([sentence])[0]
    return models.Embedding(embedding=embedding)


@app.post("/embeddings/bulk", response_model=models.Embeddings)
async def get_embeddings_bulk(sentences: models.Sentences):
    """Returns the embeddings for a list of sentences."""
    embeddings = model_client.predict(sentences.sentences)
    return models.Embeddings(embeddings=embeddings)


@app.post("/embeddings/similarity", response_model=models.Similarity)
async def get_embeddings_similarity(sentence_pair: models.SentencePair):
    """Returns the cosine similarity between a pair of sentences."""
    sentences = [sentence_pair.sentence_1, sentence_pair.sentence_2]
    embeddings = model_client.predict(sentences)
    similarity = cosine_similarity(*embeddings)
    return models.Similarity(similarity=similarity)
