from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv
import google.generativeai as genai
import uuid
from collections import defaultdict

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
    qdrant_client.recreate_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
    )
    qdrant_client.create_payload_index(
        collection_name=QDRANT_COLLECTION_NAME,
        field_name="filename",
        field_schema=models.PayloadSchemaType.KEYWORD
    )

class TextChunk(BaseModel):
    chunk: str
    filename: str
    chunk_index: int

class ChatQuery(BaseModel):
    query: str

class TranslateRequest(BaseModel):
    text: str
    language: str

class DeleteRequest(BaseModel):
    filename: str

class ChallengeRequest(BaseModel):
    statement: str

@app.post("/new-session/")
async def new_session():
    try:
        initialize_collection()
        return {"status": "success", "message": "New session started."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-chunk/")
async def upload_text_chunk(item: TextChunk):
    try:
        vector = EMBEDDING_MODEL.encode(item.chunk).tolist()
        payload = {"text": item.chunk, "filename": item.filename.strip(), "chunk_index": item.chunk_index}
        qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=[models.PointStruct(id=str(uuid.uuid4()), vector=vector, payload=payload)], wait=True)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during chunk processing: {str(e)}")

@app.post("/delete-document/")
async def delete_document(request: DeleteRequest):
    try:
        qdrant_client.delete(collection_name=QDRANT_COLLECTION_NAME, points_selector=models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="payload.filename", match=models.MatchValue(value=request.filename))])))
        return {"status": "success", "message": f"All chunks for {request.filename} have been deleted."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document chunks: {str(e)}")

@app.post("/chat/")
async def chat_with_documents(query: ChatQuery):
    try:
        query_vector = EMBEDDING_MODEL.encode(query.query).tolist()
        search_results = qdrant_client.search(collection_name=QDRANT_COLLECTION_NAME, query_vector=query_vector, limit=5)
        if not search_results: return {"answer": "Please upload a document first."}
        context = "\n---\n".join([result.payload['text'] for result in search_results])
        filenames = sorted(list(set([result.payload['filename'] for result in search_results])))
        prompt = f"Context:\n---\n{context}\n---\n\nQuestion: {query.query}\n\nAnswer based only on the context:"
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return {"answer": response.text, "context_used": filenames}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize-all/")
async def summarize_all_documents():
    try:
        all_points, _ = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10000, with_payload=True)
        if not all_points: raise HTTPException(status_code=404, detail="No documents found to summarize.")
        docs = defaultdict(list)
        for point in all_points: docs[point.payload['filename']].append(point.payload)
        individual_summaries = []
        model = genai.GenerativeModel('gemini-2.5-pro')
        for filename, chunks in docs.items():
            sorted_chunks = sorted(chunks, key=lambda c: c.get('chunk_index', 0))
            full_text = "\n".join([chunk['text'] for chunk in sorted_chunks])
            prompt = f"Provide a concise summary of the text from '{filename}':\n\n{full_text}"
            response = model.generate_content(prompt)
            individual_summaries.append(f"Summary for {filename}:\n{response.text}\n")
        if len(individual_summaries) > 1:
            all_summaries_text = "\n---\n".join(individual_summaries)
            synthesis_prompt = f"Create a single 'synthesis' summary combining key points from the following summaries:\n\n{all_summaries_text}\n\nOverall Synthesis Summary:"
            synthesis_response = model.generate_content(synthesis_prompt)
            final_summary = synthesis_response.text
        else: final_summary = individual_summaries[0]
        return {"summary": final_summary, "source_documents": list(docs.keys())}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/translate/")
async def translate_text(request: TranslateRequest):
    try:
        prompt = f"Translate the following text to {request.language}. Output only the translated text:\n\n---\n{request.text}\n---"
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return {"translated_text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/challenge/")
async def challenge_statement(request: ChallengeRequest):
    """Finds supporting and opposing arguments for a given statement."""
    try:
        supporting_vector = EMBEDDING_MODEL.encode(request.statement).tolist()
        supporting_results = qdrant_client.search(collection_name=QDRANT_COLLECTION_NAME, query_vector=supporting_vector, limit=3)
        
        opposing_query = f"arguments against, risks of, disadvantages of, or alternatives to: {request.statement}"
        opposing_vector = EMBEDDING_MODEL.encode(opposing_query).tolist()
        opposing_results = qdrant_client.search(collection_name=QDRANT_COLLECTION_NAME, query_vector=opposing_vector, limit=3)

        combined_context = {result.id: result.payload['text'] for result in supporting_results + opposing_results}
        context_text = "\n---\n".join(combined_context.values())

        if not context_text.strip():
            return {"answer": "I couldn't find any information relevant to that statement in the documents."}
        
        prompt = f"""
        You are a "Devil's Advocate" analyst. Your task is to analyze the following statement based *only* on the provided context from the documents.

        User's Statement: "{request.statement}"

        Your Task:
        1.  Scrutinize the context for points that support the user's statement.
        2.  Scrutinize the context for points that challenge, contradict, or present risks related to the statement.
        3.  Organize your findings into "Arguments For" and "Arguments Against".
        4.  If no supporting or opposing arguments are found in the context, state that clearly.
        5.  Do NOT use any external knowledge. Base your entire analysis on the text below.

        Context:
        ---
        {context_text}
        ---

        Analysis:
        """
        
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        return {"answer": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))