"""
bot/rag.py
Queries the Vertex AI RAG Engine (Corpus) and returns ranked document chunks.
Uses the vertexai.preview.rag module with the .query() style interface.
"""

import os
import logging
from dataclasses import dataclass
from typing import List

import vertexai
from vertexai.preview import rag
from vertexai.preview.rag.utils.resources import RagResource

logger = logging.getLogger("gohappy.rag")


@dataclass
class RetrievedChunk:
    index: int
    text: str
    source: str        # document title or URI
    score: float


class RAGEngine:
    """
    Thin wrapper around Vertex AI RAG Engine retrieval.

    Required env vars:
        GCP_PROJECT_ID      — your GCP project
        GCP_LOCATION        — e.g. "us-central1"
        VERTEX_RAG_CORPUS   — full resource name:
                              projects/<id>/locations/<loc>/ragCorpora/<corpus_id>
    """

    def __init__(self):
        project  = os.environ["GCP_PROJECT_ID"]
        location = os.environ.get("GCP_LOCATION", "us-central1")

        vertexai.init(project=project, location=location)

        self.corpus_name    = os.environ["VERTEX_RAG_CORPUS"]
        self.top_k          = int(os.environ.get("RAG_TOP_K", "8"))
        self.distance_threshold = float(os.environ.get("RAG_DISTANCE_THRESHOLD", "0.5"))

        logger.info(
            "RAGEngine initialised | corpus=%s  top_k=%d",
            self.corpus_name, self.top_k,
        )

    def query(self, query_text: str) -> List[RetrievedChunk]:
        """
        Retrieve the most relevant chunks from the RAG corpus for a given query.
        Returns a list of RetrievedChunk objects, ranked by relevance.
        """
        try:
            rag_resource = RagResource(rag_corpus=self.corpus_name)

            response = rag.retrieval_query(
                rag_resources=[rag_resource],
                text=query_text,
                similarity_top_k=self.top_k,
                vector_distance_threshold=self.distance_threshold,
            )

            chunks: List[RetrievedChunk] = []
            for idx, ctx in enumerate(response.contexts.contexts, start=1):
                chunks.append(
                    RetrievedChunk(
                        index=idx,
                        text=ctx.text.strip(),
                        source=ctx.source_uri or ctx.source_display_name or "GoHappy KB",
                        score=round(1.0 - ctx.distance, 4),   # distance → similarity
                    )
                )

            logger.info("RAG returned %d chunks for query: %.60s…", len(chunks), query_text)
            return chunks

        except Exception as exc:
            logger.error("RAG query failed: %s", exc, exc_info=True)
            return []   # degrade gracefully — LLM will answer from system prompt alone

    def format_for_prompt(self, chunks: List[RetrievedChunk]) -> str:
        """
        Serialize the retrieved chunks into a labeled block
        ready to be injected into the LLM prompt.
        """
        if not chunks:
            return "(No relevant documents retrieved from the knowledge base.)"

        parts = []
        for chunk in chunks:
            parts.append(
                f"[DOC_{chunk.index}] (source: {chunk.source}, score: {chunk.score})\n"
                f"{chunk.text}"
            )
        return "\n\n".join(parts)
