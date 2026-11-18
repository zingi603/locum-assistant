"""Embedding models. FastEmbedder runs BAAI/bge-small-en-v1.5 locally through ONNX: no API key,
no per-call cost, 384-dimensional vectors. HashingEmbedder is a deterministic stand-in for tests,
so the test suite never downloads a model."""

import hashlib
import math
import re
from collections.abc import Iterable
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class FastEmbedder:
    def __init__(self, model_name: str, dim: int) -> None:
        from fastembed import TextEmbedding  # imported here: loading the model is slow

        self.dim = dim
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        # bge models expect queries to carry an instruction prefix; query_embed adds it
        return next(iter(self._model.query_embed(text))).tolist()


class HashingEmbedder:
    """Bag of words hashed into a fixed-size vector, L2-normalised. Texts that share words get
    similar vectors, which is all the tests need."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def _vector(self, words: Iterable[str]) -> list[float]:
        v = [0.0] * self.dim
        for w in words:
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(re.findall(r"[a-z0-9]+", text.lower()))
