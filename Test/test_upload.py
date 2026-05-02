import os
import traceback
import vertexai
from vertexai.preview import rag
from dotenv import load_dotenv

load_dotenv()
project = os.environ["GCP_PROJECT_ID"]
location = os.environ.get("GCP_LOCATION", "us-central1")
corpus_name = os.environ["VERTEX_RAG_CORPUS"]

vertexai.init(project=project, location=location)

print("Uploading...")
try:
    rag.upload_file(
        corpus_name=corpus_name,
        path="/Users/bhumitgoyal/Downloads/GoHappyClub_KnowledgeBase.md",
        display_name="GoHappyClub_KnowledgeBase.md"
    )
    print("Success!")
except Exception as e:
    traceback.print_exc()
