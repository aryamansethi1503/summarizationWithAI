from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import google.generativeai as genai
import uuid

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
   raise ValueError("GEMINI_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION_NAME = "my-gemini-docs-chunked"

qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_collection():
   """Wipes and recreates the collection and its index."""
   print("Clearing and recreating Qdrant collection for a new session.")
   qdrant_client.recreate_collection(
       collection_name=QDRANT_COLLECTION_NAME,
       vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
   )
   qdrant_client.create_payload_index(
       collection_name=QDRANT_COLLECTION_NAME,
       field_name="filename",
       field_schema=models.PayloadSchemaType.KEYWORD
   )
   print(f"Collection '{QDRANT_COLLECTION_NAME}' is ready.")

class TextChunk(BaseModel):
   chunk: str
   filename: str
   chunk_index: int

class ChatQuery(BaseModel):
   query: str

class SummarizeRequest(BaseModel):
   filename: str

@app.post("/new-session/")
async def new_session():
   """Clears the database for a new session."""
   try:
       initialize_collection()
       return {"status": "success", "message": "New session started. Database is clear."}
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-chunk/")
async def upload_text_chunk(item: TextChunk):
   """Receives and stores a single chunk of text."""
   try:
       vector = EMBEDDING_MODEL.encode(item.chunk).tolist()
       point_id = str(uuid.uuid4())
       payload = {"text": item.chunk, "filename": item.filename.strip(), "chunk_index": item.chunk_index}
       qdrant_client.upsert(
           collection_name=QDRANT_COLLECTION_NAME,
           points=[models.PointStruct(id=point_id, vector=vector, payload=payload)],
           wait=True,
       )
       return {"status": "success"}
   except Exception as e:
       raise HTTPException(status_code=500, detail=f"An error occurred during chunk processing: {str(e)}")

@app.post("/chat/")
async def chat_with_documents(query: ChatQuery):
   try:
       query_vector = EMBEDDING_MODEL.encode(query.query).tolist()
       search_results = qdrant_client.search(collection_name=QDRANT_COLLECTION_NAME, query_vector=query_vector, limit=5)
       if not search_results: return {"answer": "The document has not been processed. Please upload a document first."}
       context = "\n---\n".join([result.payload['text'] for result in search_results])
       filenames = sorted(list(set([result.payload['filename'] for result in search_results])))
       prompt = f"Context:\n---\n{context}\n---\n\nQuestion: {query.query}\n\nAnswer based only on the context:"
       model = genai.GenerativeModel('gemini-2.5-pro')
       response = model.generate_content(prompt)
       return {"answer": response.text, "context_used": filenames}
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize/")
async def summarize_document(request: SummarizeRequest):
   try:
       clean_filename = request.filename.strip()
       all_points, _ = qdrant_client.scroll(
           collection_name=QDRANT_COLLECTION_NAME,
           scroll_filter=models.Filter(must=[models.FieldCondition(key="payload.filename", match=models.MatchValue(value=clean_filename))]),
           limit=10000, with_payload=True,
       )
       if not all_points:
           raise HTTPException(status_code=404, detail=f"No content found for document: {request.filename}. Please upload it first.")
       sorted_points = sorted(all_points, key=lambda p: p.payload.get('chunk_index', 0))
       full_text = "\n".join([point.payload['text'] for point in sorted_points])
       if not full_text.strip(): return {"summary": "Document appears to be empty or contains no extractable text."}
       prompt = f"Provide a concise summary of the following text from the document '{request.filename}':\n\n{full_text}"
       model = genai.GenerativeModel('gemini-2.5-pro')
       response = model.generate_content(prompt)
       return {"summary": response.text, "filename": request.filename}
   except Exception as e:
       raise HTTPException(status_code=500, detail=str(e))
