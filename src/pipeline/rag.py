"""RAG pipeline — chunk, embed, index, retrieve, generate.

Reaproveita as funcoes do notebook 02.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.utils.embedding_functions import (
    OpenAIEmbeddingFunction,
    GoogleGeminiEmbeddingFunction,
    GoogleGenerativeAiEmbeddingFunction,
)
from openai import OpenAI
import uuid
import time
import re
import random
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class _EmbeddingWrapper:
    """Wrapper que tenta a função primária e faz fallback para a secundária em caso de 404/not found.

    Ao detectar erro de modelo não encontrado, tenta o fallback e o promove para primário.
    """

    def __init__(self, primary, fallback=None, model_name: str | None = None):
        self.primary = primary
        self.fallback = fallback
        self.model_name = model_name or getattr(primary, "model_name", None)

    def __call__(self, input):
        # primeira tentativa com o primário
        try:
            return self.primary(input)
        except Exception as e:
            msg = str(e).lower()
            # detectar erros de modelo nao encontrado
            if self.fallback and ("not found" in msg or "is not found" in msg or "404" in msg):
                try:
                    result = self.fallback(input)
                    # promover fallback para primário para chamadas futuras
                    self.primary = self.fallback
                    return result
                except Exception:
                    # se fallback também falhar, relançar o erro original
                    raise
            # caso não seja um erro tratável, relançar
            raise

    # API compatibility for chromadb embedding function
    def name(self) -> str:
        try:
            return self.primary.name()
        except Exception:
            return getattr(self, "model_name", "wrapped")

    def default_space(self):
        try:
            return self.primary.default_space()
        except Exception:
            return "cosine"

    def supported_spaces(self):
        try:
            return self.primary.supported_spaces()
        except Exception:
            return ["cosine", "l2", "ip"]

    def get_config(self):
        cfg = {}
        try:
            cfg = self.primary.get_config() or {}
        except Exception:
            pass
        # expose model_name for diagnostics
        if "model_name" not in cfg and self.model_name:
            cfg["model_name"] = self.model_name
        return cfg

    def __getattr__(self, item):
        # Delegate attribute access to primary for full compatibility
        return getattr(self.primary, item)



def _make_client() -> tuple[OpenAI | None, str | None]:
    """Inicializa cliente OpenAI-compatible conforme provider escolhido no .env."""
    if "GEMINI_API_KEY" in os.environ:
        client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
        embed_api_base = "https://generativelanguage.googleapis.com/v1beta/openai/"
    elif "OPENAI_API_KEY" in os.environ:
        client = OpenAI()
        embed_api_base = None
    else:
        # No API key configured — return None client and allow the rest of the
        # pipeline to be used for offline tasks (indexing) or local embedding
        # fallbacks. Calling LLM APIs will raise a clear error later.
        return None, None
    return client, embed_api_base


def _make_embed_fn() -> OpenAIEmbeddingFunction:
    # Prefer Google/ Gemini native embedding functions when a GEMINI_API_KEY is present
    if "GEMINI_API_KEY" in os.environ:
        gemini_model = (
            os.environ.get("GEMINI_EMBED_MODEL")
            or os.environ.get("EMBED_MODEL")
            or "embedding-1"
        )
        # If user selected a "text-embedding-*" model (Gemini native), use the Google embedding helper
        if gemini_model.startswith("text-embedding"):
            try:
                primary = GoogleGenerativeAiEmbeddingFunction(
                    api_key=os.environ.get("GEMINI_API_KEY"),
                    model_name=gemini_model,
                )
                return _EmbeddingWrapper(primary=primary, fallback=None, model_name=gemini_model)
            except Exception:
                # Fall back to GoogleGeminiEmbeddingFunction if available
                try:
                    primary = GoogleGeminiEmbeddingFunction(model_name=gemini_model)
                    return _EmbeddingWrapper(primary=primary, fallback=None, model_name=gemini_model)
                except Exception:
                    raise
        # Otherwise use OpenAI-compatible endpoint for embedding-1 and similar models
        # Try OpenAI-compatible client first, but prepare a Google native fallback
        primary = OpenAIEmbeddingFunction(
            api_key=os.environ["GEMINI_API_KEY"],
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            model_name=gemini_model,
        )
        # prepare fallback (may raise on construction if dependencies missing)
        try:
            fallback = GoogleGenerativeAiEmbeddingFunction(
                api_key=os.environ.get("GEMINI_API_KEY"), model_name=gemini_model
            )
        except Exception:
            try:
                fallback = GoogleGeminiEmbeddingFunction(model_name=gemini_model)
            except Exception:
                fallback = None

        return _EmbeddingWrapper(primary=primary, fallback=fallback, model_name=gemini_model)
    elif "OPENAI_API_KEY" in os.environ:
        return OpenAIEmbeddingFunction(
            api_key=os.environ["OPENAI_API_KEY"],
            model_name="text-embedding-3-small",
        )
    else:
        # Fallback: Ollama local — apenas para desenvolvimento local
        print(
            "[AVISO] Nenhuma API key encontrada. Usando Ollama local como fallback. "
            "Certifique-se de ter `ollama pull nomic-embed-text` executado."
        )
        return OpenAIEmbeddingFunction(
            api_key="ollama",
            api_base="http://localhost:11434/v1",
            model_name="nomic-embed-text",
        )


class RAGPipeline:
    """Pipeline RAG end-to-end com Chroma local."""

    def __init__(
        self,
        corpus_dir: str = "data/corpus",
        persist_dir: str = "data/chroma",
        collection_name: str = "docs",
        llm_model: str | None = None,
        embed_model: str | None = None,  # reservado para override futuro
    ) -> None:
        self.client, _ = _make_client()
        self.llm_model = llm_model or os.environ.get("LLM_MODEL", "gemini-3.1-flash-lite")
        self.embed_fn = _make_embed_fn()

        # Startup validation: call the selected embedding function directly to validate configuration
        model_name = getattr(
            self.embed_fn,
            "model_name",
            os.environ.get("GEMINI_EMBED_MODEL") or os.environ.get("OPENAI_EMBED_MODEL") or os.environ.get("EMBED_MODEL") or "unknown",
        )
        try:
            # embedding functions expect a list of texts
            self.embed_fn(["teste"])
        except Exception as e:
            raise RuntimeError(
                f"Embedding model test failed for model={model_name}: {e}\n"
                "Check the available models or set GEMINI_EMBED_MODEL/OPENAI_EMBED_MODEL correctly."
            ) from e

        self.corpus_dir = Path(corpus_dir)
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        chroma = chromadb.PersistentClient(path=persist_dir)
        # cast to Any to satisfy chroma typing (embedding function typing is permissive at runtime)
        self.collection = chroma.get_or_create_collection(
            name=collection_name, embedding_function=cast(Any, self.embed_fn)
        )

    # ------------------------------------------------------------------ TODO 1
    def ingest_and_index(self) -> int:
        """Le PDFs de `corpus_dir`, faz chunking e indexa em Chroma."""

        # TODO 1.A: Ler PDFs e extrair paginas
        docs: list[dict] = []
        pdf_files = list(self.corpus_dir.glob("*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(
                f"Nenhum PDF encontrado em '{self.corpus_dir}'. "
                "Adicione seus PDFs na pasta data/corpus/ antes de indexar."
            )

        for filepath in pdf_files:
            reader = PdfReader(filepath)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                # CORRECAO: ignorar paginas em branco antes de adicionar
                if text.strip():
                    docs.append({
                        "text": text,
                        "source": filepath.name,
                        "page": i + 1,
                    })

        if not docs:
            raise ValueError(
                "Nenhum texto extraido dos PDFs. "
                "Verifique se os arquivos possuem camada de texto (nao sao scans)."
            )

        # TODO 1.B: Chunking Recursivo
        chunks: list[dict] = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
        )

        for doc in docs:
            split_texts = text_splitter.split_text(doc["text"])
            for text_chunk in split_texts:
                # CORRECAO: ignorar chunks vazios apos split
                if text_chunk.strip():
                    chunks.append({
                        "id": str(uuid.uuid4()),
                        "text": text_chunk,
                        "source": doc["source"],
                        "page": doc["page"],
                    })

        # TODO 1.C: Adicionar ao Chroma em batches (limite de 100 por batch)
        if chunks:
            batch_size = 100
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                # Try adding with retries to handle rate limits / quota errors
                max_attempts = 6
                for attempt in range(1, max_attempts + 1):
                    try:
                        self.collection.add(
                            ids=[c["id"] for c in batch],
                            documents=[c["text"] for c in batch],
                            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
                        )
                        break
                    except Exception as e:
                        # Try to parse a retry delay from the error message
                        msg = str(e)
                        m = re.search(r"retry(?:Delay)?[:=\s]*'?([0-9]+)s|'?Please retry in ([0-9.]+)s", msg)
                        if m:
                            # pick the first non-empty capture group
                            s = next(g for g in m.groups() if g)
                            try:
                                wait = float(s) + 1.0
                            except Exception:
                                wait = min(60, 2 ** attempt)
                        else:
                            # exponential backoff with jitter
                            wait = min(60, (2 ** attempt) + random.random())

                        if attempt == max_attempts:
                            raise

                        print(f"[ingest] Rate limit or transient error on batch add (attempt {attempt}/{max_attempts}): {e}. Waiting {wait:.1f}s and retrying...")
                        time.sleep(wait)

        print(f"[ingest] {len(pdf_files)} PDFs | {len(docs)} paginas | {len(chunks)} chunks indexados.")
        return self.collection.count()

    # ------------------------------------------------------------------ TODO 2
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Busca top-k chunks similares a query."""

        # CORRECAO: verificar se a collection tem documentos antes de buscar
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),  # n_results nao pode exceder o total
        )

        hits: list[dict] = []

        docs_lists = results.get("documents")
        if not docs_lists:
            return hits

        docs = docs_lists[0] if len(docs_lists) > 0 else []
        metadatas_lists = results.get("metadatas") or []
        metadatas = metadatas_lists[0] if len(metadatas_lists) > 0 else ([{}] * len(docs))
        distances_lists = results.get("distances") or []
        distances = distances_lists[0] if len(distances_lists) > 0 else ([0.0] * len(docs))

        for doc, meta, dist in zip(docs, metadatas, distances):
            hits.append({
                "text": doc,
                "source": meta.get("source", "desconhecido"),
                "page": meta.get("page", 0),
                "distance": dist,
            })

        return hits

    # ------------------------------------------------------------------ TODO 3
    def answer(self, question: str, k: int = 5) -> dict:
        """Pipeline completo: retrieve + augment + generate. Retorna {answer, sources}."""
        hits = self.retrieve(question, k=k)

        # CORRECAO: retornar mensagem util quando nao ha documentos indexados
        if not hits:
            return {
                "answer": (
                    "Nenhum documento foi encontrado no corpus. "
                    "Execute `pipeline.ingest_and_index()` antes de fazer perguntas."
                ),
                "sources": [],
            }

        # 1. Montar o contexto com cabecalhos de fonte
        context_parts = []
        for hit in hits:
            context_parts.append(f"[{hit['source']}:p{hit['page']}]\n{hit['text']}")
        context_str = "\n\n".join(context_parts)

        # 2. Construir o prompt
        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)

        # 3. Chamar a API
        try:
            if self.client is None:
                raise RuntimeError(
                    "Nenhuma API key configurada para chamadas LLM. Configure GEMINI_API_KEY ou OPENAI_API_KEY."
                )
            response = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            answer_text = response.choices[0].message.content or "Sem resposta da API."
        except Exception as e:
            answer_text = f"Erro ao chamar o LLM: {e}"

        # 4. Retornar removendo fontes duplicadas
        unique_sources = list({(h["source"], h["page"]) for h in hits})

        return {"answer": answer_text, "sources": unique_sources}


PROMPT_TEMPLATE = """Voce e um assistente tecnico especializado em LGPD e protecao de dados.
Responda APENAS com base no contexto abaixo.
Se a informacao nao estiver no contexto, diga "Nao encontrado no corpus".
Sempre cite a fonte usando o formato [arquivo:pagina].

CONTEXTO:
{context}

PERGUNTA: {question}

RESPOSTA:"""


def build_rag_pipeline(corpus_dir: str = "data/corpus") -> RAGPipeline:
    """Factory: cria pipeline e indexa corpus se ainda nao indexado."""
    pipeline = RAGPipeline(corpus_dir=corpus_dir)
    if pipeline.collection.count() == 0:
        pipeline.ingest_and_index()
    return pipeline
