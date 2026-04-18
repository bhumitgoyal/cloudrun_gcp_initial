"""
test_rag.py
Standalone script to test RAG retrieval + Gemini answer generation
without WhatsApp or Firestore dependencies.

Usage:
    set -a && source .env && set +a
    python test_rag.py
"""

import os
import asyncio
import logging

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger("test_rag")


def test_rag_retrieval(query: str):
    """Test RAG retrieval only — no Gemini, no Firestore."""
    from bot.rag import RAGEngine

    logger.info("Initialising RAG Engine...")
    rag = RAGEngine()

    logger.info("Querying: %s", query)
    chunks = rag.query(query)

    if not chunks:
        logger.warning("No chunks returned!")
        return

    print("\n" + "=" * 70)
    print(f"  RAG RESULTS — {len(chunks)} chunks for: \"{query}\"")
    print("=" * 70)
    for chunk in chunks:
        print(f"\n[DOC_{chunk.index}]  score={chunk.score}  source={chunk.source}")
        print("-" * 50)
        print(chunk.text[:500])
        if len(chunk.text) > 500:
            print(f"  ... ({len(chunk.text)} chars total)")
    print("=" * 70 + "\n")

    return chunks


async def test_rag_plus_gemini(query: str):
    """Test full RAG → Gemini pipeline (no WhatsApp, no Firestore)."""
    from bot.rag import RAGEngine
    from bot.llm import GeminiChat

    logger.info("Initialising RAG Engine...")
    rag = RAGEngine()

    logger.info("Initialising Gemini...")
    llm = GeminiChat()

    # 0. Rewrite query
    logger.info("Polishing query...")
    polished_query = await llm.rewrite_query(query)
    print(f"\n✨ Polished query: {polished_query}\n")

    # 1. RAG retrieval
    logger.info("Querying RAG: %s", polished_query)
    chunks = rag.query(polished_query)
    retrieved_context = rag.format_for_prompt(chunks)

    print(f"\n📚 Retrieved {len(chunks)} chunks from RAG corpus\n")

    # 2. Call Gemini with the retrieved context
    logger.info("Calling Gemini with retrieved context...")
    response = await llm.chat(
        customer_summary="New member. This is the start of the conversation — no prior context.",
        conversation_history="(No prior conversation history.)",
        user_query=query,
        retrieved_context=retrieved_context,
    )

    print("=" * 70)
    print("  GEMINI RESPONSE")
    print("=" * 70)
    print(f"\n💬 Answer: {response.answer}")
    print(f"🚨 Escalation: {response.escalation}")
    print("=" * 70 + "\n")

    return response


def interactive_mode():
    """Interactive REPL for testing queries."""
    print("\n" + "=" * 70)
    print("  GoHappy RAG Test Console")
    print("  Type a query and press Enter. Type 'quit' to exit.")
    print("  Prefix with 'rag:' for RAG-only (no Gemini).")
    print("=" * 70 + "\n")

    while True:
        try:
            query = input("🔍 Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not query or query.lower() == "quit":
            print("Bye!")
            break

        if query.lower().startswith("rag:"):
            # RAG-only mode
            test_rag_retrieval(query[4:].strip())
        else:
            # Full RAG + Gemini
            asyncio.run(test_rag_plus_gemini(query))


if __name__ == "__main__":
    # Quick single-query test if TEST_QUERY env var is set, otherwise interactive
    test_query = os.environ.get("TEST_QUERY")
    if test_query:
        print(f"\nRunning single test query: {test_query}\n")
        chunks = test_rag_retrieval(test_query)
        if chunks:
            asyncio.run(test_rag_plus_gemini(test_query))
    else:
        interactive_mode()
