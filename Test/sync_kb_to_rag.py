"""
Test/sync_kb_to_rag.py
One-shot script to sync the local knowledge base file to Vertex AI RAG Corpus.

Steps:
  1. Lists all existing files in the RAG corpus
  2. Deletes them (to avoid stale content)
  3. Uploads the new gohappy_club_knowledge_base.md
"""

import os
import sys
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set env from .env if not already set
from pathlib import Path
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())

import vertexai
from vertexai.preview import rag

PROJECT = os.environ["GCP_PROJECT_ID"]
LOCATION = os.environ.get("GCP_LOCATION", "asia-south1")
CORPUS = os.environ["VERTEX_RAG_CORPUS"]
KB_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "gohappy_club_knowledge_base.md")


def main():
    print(f"Project:  {PROJECT}")
    print(f"Location: {LOCATION}")
    print(f"Corpus:   {CORPUS}")
    print(f"KB File:  {KB_FILE}")
    print()

    vertexai.init(project=PROJECT, location=LOCATION)

    # 1. List existing files
    print("── Listing existing RAG files ──")
    existing_files = list(rag.list_files(corpus_name=CORPUS))
    if existing_files:
        for f in existing_files:
            print(f"  • {f.name}  ({f.display_name})")
    else:
        print("  (no existing files)")
    print()

    # 2. Delete old files
    if existing_files:
        print("── Deleting old files ──")
        for f in existing_files:
            print(f"  Deleting {f.display_name} ...", end=" ")
            rag.delete_file(name=f.name)
            print("✅")
        print()

    # 3. Upload new KB
    print("── Uploading new knowledge base ──")
    file_size = os.path.getsize(KB_FILE)
    print(f"  File size: {file_size:,} bytes")

    for attempt in range(3):
        try:
            result = rag.upload_file(
                corpus_name=CORPUS,
                path=KB_FILE,
                display_name="GoHappyClub_KnowledgeBase.md",
            )
            print(f"  ✅ Upload successful!")
            print(f"  Resource name: {result.name}")
            break
        except Exception as e:
            print(f"  ⚠️  Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                print("  Retrying in 3s...")
                time.sleep(3)
            else:
                print("  ❌ All attempts failed.")
                sys.exit(1)

    # 4. Verify
    print()
    print("── Verifying upload ──")
    final_files = list(rag.list_files(corpus_name=CORPUS))
    for f in final_files:
        print(f"  • {f.name}  ({f.display_name})")
    print()
    print(f"✅ Done! {len(final_files)} file(s) in corpus.")


if __name__ == "__main__":
    main()
