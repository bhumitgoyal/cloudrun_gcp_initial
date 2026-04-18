"""
Debug: Show Gemini rewrite output + cosine similarity for each pair
using the newly updated canonical LLM prompt.
"""
import os, asyncio
from dotenv import load_dotenv
load_dotenv()

os.environ["CACHE_SIMILARITY_THRESHOLD"] = "0.75"
import logging
logging.basicConfig(level=logging.WARNING)

from bot.llm import GeminiChat
from bot.rag_cache import RAGCache
import numpy as np

PAIRS = [
    ("How do I join GoHappy Club?",                    "membership kaise leni hai GoHappy ki"),
    ("What are the different membership plans?",       "kitne type ke plans hain aapke"),
    ("How much does the Gold plan cost?",              "gold plan ka price kya hai"),
    ("What sessions are available this week?",         "is week kaunsi classes hain"),
    ("What time does the yoga class start?",           "yoga class ka time kya hai"),
    ("What are Happy Coins?",                          "happy coins kya hote hain"),
    ("How do I download the GoHappy app?",             "app kaise download kare"),
    ("How do I earn Happy Coins?",                     "happy coins kaise milte hain"),
    ("What trips are coming up?",                      "koi trip aa raha hai kya"),
    ("How much does the Shimla trip cost?",            "shimla trip kitne ka hai"),
    ("Can I pay using UPI?",                           "UPI se payment ho jayegi kya"),
    ("How do I contact customer support?",             "customer care ka number do"),
    ("How do I cancel my membership?",                 "plan cancel karna hai mujhe"),
    ("What is GoHappy Club?",                          "GoHappy Club ke baare mein batao"),
    ("Is GoHappy Club available in my city?",          "kya GoHappy Pune mein available hai"),
]

async def main():
    gemini = GeminiChat()
    cache = RAGCache()

    await cache._ensure_model()

    print(f"\n{'─'*130}")
    print(f"  {'Phase1 (cached as)':<40}  {'Phase2 (Hinglish)':<35}  {'Gemini rewrote to':<40}  {'Sim':>5}  Hit?")
    print(f"{'─'*130}")

    hits = 0
    for p1, p2 in PAIRS:
        rewritten = await gemini.rewrite_query(p2)
        emb1 = cache._model.encode([p1])[0]
        emb2 = cache._model.encode([rewritten])[0]
        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
        hit = sim >= float(os.environ["CACHE_SIMILARITY_THRESHOLD"])
        if hit: hits += 1
        mark = "\033[92m✅\033[0m" if hit else "\033[91m❌\033[0m"
        print(f"  {p1:<40}  {p2:<35}  {rewritten:<40}  {sim:.3f}  {mark}")
        # Sleep to avoid 429
        await asyncio.sleep(1)

    print(f"{'─'*130}")
    print(f"  Hits at threshold {os.environ['CACHE_SIMILARITY_THRESHOLD']}: {hits}/{len(PAIRS)} ({hits/len(PAIRS)*100:.0f}%)\n")

if __name__ == "__main__":
    asyncio.run(main())
