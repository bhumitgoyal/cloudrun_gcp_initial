import os
import vertexai
from vertexai.preview import rag
from dotenv import load_dotenv

load_dotenv()
project = os.environ["GCP_PROJECT_ID"]
location = os.environ.get("GCP_LOCATION", "us-central1")
corpus_name = os.environ["VERTEX_RAG_CORPUS"]

vertexai.init(project=project, location=location)

files = list(rag.list_files(corpus_name=corpus_name))
if files:
    f = files[0]
    print(dir(f))
