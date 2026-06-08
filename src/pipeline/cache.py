"""Cache em 2 niveis: exact-match (SHA256) + semantic (cosine similarity).

Reaproveita o notebook 05.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any

import numpy as np
from openai import OpenAI


class ExactCache:
    """Cache por hash SHA256 da query. Captura replays exatos (~10-15% das queries)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(query: str) -> str:
        return hashlib.sha256(query.encode()).hexdigest()

    def get(self, query: str) -> str | None:
        return self._store.get(self._key(query))

    def put(self, query: str, answer: str) -> None:
        self._store[self._key(query)] = answer

    def stats(self) -> dict[str, int]:
        return {"size": len(self._store)}


class SemanticCache:
    """Cache por similaridade de embedding. Captura parafrases (~20% adicional)."""

    def __init__(self, threshold: float = 0.93) -> None:
        self.threshold = threshold
        self._queries: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._answers: list[str] = []

        # CORRECAO: provider de embedding alinhado com o resto do projeto.
        # Se GEMINI_API_KEY estiver disponivel, usa Gemini embeddings.
        # Caso contrario, tenta Ollama local como fallback.
        if "GEMINI_API_KEY" in os.environ:
            self._client = OpenAI(
                api_key=os.environ["GEMINI_API_KEY"],
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            )
            self._embed_model = os.environ.get("GEMINI_EMBED_MODEL", "embedding-1")
            # Try a quick embedding to warn early if model is not available
            try:
                self._client.embeddings.create(model=self._embed_model, input="teste")
            except Exception as e:
                print(
                    f"[AVISO] Falha ao validar modelo de embedding Gemini={self._embed_model}: {e}. "
                    "Verifique GEMINI_EMBED_MODEL ou use OPENAI_API_KEY como fallback."
                )
        else:
            # Fallback: Ollama local (requer `ollama pull nomic-embed-text`)
            self._client = OpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1/",
            )
            self._embed_model = "nomic-embed-text"

    def _embed(self, text: str) -> np.ndarray:
        r = self._client.embeddings.create(model=self._embed_model, input=text)
        return np.array(r.data[0].embedding)

    # ------------------------------------------------------------------ TODO 5
    def get(self, query: str) -> str | None:
        """Retorna resposta cacheada se similar a query alguma anterior, OU None."""
        if not self._queries:
            return None

        # 1. Embedar a nova query
        query_emb = self._embed(query)

        best_sim = -1.0
        best_idx = -1

        # 2. Calcular similaridade cosseno contra todos os embeddings salvos
        for i, emb in enumerate(self._embeddings):
            norm_q = np.linalg.norm(query_emb)
            norm_e = np.linalg.norm(emb)

            # CORRECAO: proteger contra divisao por zero
            if norm_q == 0 or norm_e == 0:
                continue

            sim = float(np.dot(query_emb, emb) / (norm_q * norm_e))

            if sim > best_sim:
                best_sim = sim
                best_idx = i

        # 3. Verificar se atinge o threshold
        if best_idx >= 0 and best_sim >= self.threshold:
            return self._answers[best_idx]

        # 4. Caso contrario: cache miss
        return None

    def put(self, query: str, answer: str) -> None:
        self._queries.append(query)
        self._embeddings.append(self._embed(query))
        self._answers.append(answer)

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._queries), "threshold": self.threshold}
