"""Model routing cheap-first com fallback.

Reaproveita o notebook 05.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True)
class RouteDecision:
    model: str
    complexity: str  # "simple" | "complex"
    reason: str


# ------------------------------------------------------------------ TODO 6
def classify_complexity(query: str) -> RouteDecision:
    """
    Classifica complexidade da query para escolher modelo (cheap vs premium).

    CORRECAO: versao original usava apenas keywords hardcoded, o que nao captura
    perguntas complexas escritas de forma direta (ex: "O que e dado sensivel?").
    Nova versao combina heuristicas com um fallback confiavel:

    Regra 1 — comprimento longo (>150 chars) -> complex
    Regra 2 — presenca de palavras-chave de analise -> complex
    Regra 3 — perguntas curtas e diretas -> simple

    O modelo 'cheap' e suficiente para lookup de artigo ou pergunta factual simples.
    O modelo 'premium' e necessario para comparacoes, analises e cenarios compostos.
    """
    cheap_model = os.environ.get("CHEAP_MODEL", "gemini-2.0-flash")
    premium_model = os.environ.get("PREMIUM_MODEL", "gemini-2.5-pro")

    query_lower = query.lower().strip()

    # --- Heuristica 1: comprimento como proxy de complexidade ---
    if len(query) > 150:
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason="Query longa (>150 chars) — provavelmente requer analise aprofundada.",
        )

    # --- Heuristica 2: palavras-chave que indicam analise ou comparacao ---
    complex_keywords = [
        "explique", "explica", "explica-me",
        "compare", "comparar", "diferenca", "diferenca entre", "versus", " vs ",
        "analise", "analisa", "analisar",
        "resuma", "resumo",
        "quando posso", "quando e possivel", "e possivel",
        "quais sao", "quais os", "liste",
        "como funciona", "como tratar", "como implementar",
        "o que acontece se", "quais as consequencias",
        "exemplo", "exemplos",
        "cenario", "situacao",
        "posso armazenar", "posso coletar", "posso compartilhar",
        "base legal", "hipotese", "hipoteses",
    ]

    if any(kw in query_lower for kw in complex_keywords):
        matched = [kw for kw in complex_keywords if kw in query_lower]
        return RouteDecision(
            model=premium_model,
            complexity="complex",
            reason=f"Palavras-chave de analise detectadas: {matched}.",
        )

    # --- Heuristica 3: perguntas curtas e diretas -> modelo barato ---
    return RouteDecision(
        model=cheap_model,
        complexity="simple",
        reason="Query curta e direta — modelo rapido suficiente.",
    )


def make_client() -> OpenAI:
    """Cliente OpenAI-compatible para o provider configurado."""
    if "GEMINI_API_KEY" in os.environ:
        return OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    return OpenAI()
