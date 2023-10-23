from pydantic import BaseModel


class Embedding(BaseModel):
    """Represents a single vector embedding."""

    embedding: list[float]


class Embeddings(BaseModel):
    """Represents a collection vector embeddings."""

    embeddings: list[list[float]]


class Sentences(BaseModel):
    """Represents a collection of sentences."""

    sentences: list[str]


class SentencePair(BaseModel):
    """Represents a pair of sentences."""

    sentence_1: str
    sentence_2: str


class Similarity(BaseModel):
    """Reprensent similarity score."""

    similarity: float
