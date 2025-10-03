import streamlit as st
import requests
import os
from PyPDF2 import PdfReader
from docx import Document

ORCHESTRATOR_URL = "http://127.0.0.1:8000"
OCR_URL = "http://127.0.0.1:8002/ocr/"
CHUNK_SIZE_CHARS = 2000

def extract_text_from_pdf(file):
   try:
       pdf_reader = PdfReader(file)
       text = ""
       for page in pdf_reader.pages:
           page_text = page.extract_text()
           if page_text:
               text += page_text + "\n"
       return text
   except Exception as e:
       st.error(f"Error reading PDF file: {e}")
       return None

def extract_text_from_image(file):
   try:
       files = {'file': (file.name, file, file.type)}
       response = requests.post(OCR_URL, files=files)
       response.raise_for_status()
       return response.json()["text"]
   except requests.RequestException as e:
       st.error(f"Error contacting OCR service: {e}")
       return None

def extract_text_from_docx(file):
   try:
       doc = Document(file)
       full_text = []
       for para in doc.paragraphs:
           full_text.append(para.text)
       return "\n".join(full_text)
   except Exception as e:
       st.error(f"Error reading DOCX file: {e}")
       return None

def extract_text_from_txt(file):
   try:
       return file.getvalue().decode("utf-8")
   except Exception as e:
       st.error(f"Error reading TXT file: {e}")
       return None

st.set_page_config(layout="wide")
st.title("DocQuery AI: Chat & Summarize")

if "processed_filename" not in st.session_state:
   st.session_state.processed_filename = None
if "messages" not in st.session_state:
   st.session_state.messages = []

with st.sidebar:
   st.header("Start New Session")
   if st.button("Clear Database and Start New", use_container_width=True):
       with st.spinner("Starting new session..."):
           try:
               requests.post(f"{ORCHESTRATOR_URL}/new-session/")
               st.session_state.processed_filename = None
               st.session_state.messages = []
               st.success("Ready for new document.")
               st.rerun()
           except requests.ConnectionError:
               st.error("Connection failed. Is the orchestrator running?")

   st.divider()
   st.header("Upload Document")
   uploaded_file = st.file_uploader(
       "Upload a PDF, DOCX, TXT, or image file",
       type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'docx']
   )

   if uploaded_file and (uploaded_file.name != st.session_state.get('processed_filename')):
       with st.spinner(f"Processing {uploaded_file.name}..."):
           text = None
           file_ext = os.path.splitext(uploaded_file.name)[-1].lower()

           if file_ext == ".pdf":
               text = extract_text_from_pdf(uploaded_file)
           elif file_ext in [".png", ".jpg", ".jpeg"]:
               text = extract_text_from_image(uploaded_file)
           elif file_ext == ".txt":
               text = extract_text_from_txt(uploaded_file)
           elif file_ext == ".docx":
               text = extract_text_from_docx(uploaded_file)
          
           if text:
               chunks = [text[i:i + CHUNK_SIZE_CHARS] for i in range(0, len(text), CHUNK_SIZE_CHARS)]
               total_chunks = len(chunks)
              
               progress_bar = st.progress(0, text=f"Indexing document ({total_chunks} chunks)...")
              
               upload_successful = True
               for i, chunk in enumerate(chunks):
                   try:
                       payload = {"chunk": chunk, "filename": uploaded_file.name, "chunk_index": i}
                       response = requests.post(f"{ORCHESTRATOR_URL}/upload-chunk/", json=payload)
                       if response.status_code != 200:
                           st.error(f"Error on chunk {i+1}: {response.text}")
                           upload_successful = False; break
                       progress_bar.progress((i + 1) / total_chunks, text=f"Indexing chunk {i+1}")
                   except requests.ConnectionError:
                       st.error("Connection failed. Is the orchestrator running?"); upload_successful = False; break
              
               if upload_successful:
                   st.success(f"**{uploaded_file.name}** is ready!")
                   st.session_state.processed_filename = uploaded_file.name
                   st.session_state.messages = []
                   st.rerun()
           else:
               st.error("Could not extract text from the document.")
               st.session_state.processed_filename = None

   st.divider()
   st.header("Actions")

   if st.session_state.processed_filename:
       st.info(f"Active Document: **{st.session_state.processed_filename}**")
       if st.button("Generate Summary", use_container_width=True, type="primary"):
           with st.spinner("Generating summary..."):
               try:
                   payload = {"filename": st.session_state.processed_filename}
                   response = requests.post(f"{ORCHESTRATOR_URL}/summarize/", json=payload)
                   if response.status_code == 200:
                       summary = response.json().get("summary")
                       st.session_state.messages.insert(0, {"role": "assistant", "content": f"**Summary of *{st.session_state.processed_filename}*:**\n\n{summary}"})
                   else:
                       st.error(f"Error during summarization: {response.status_code} - {response.text}")
               except requests.ConnectionError:
                   st.error("Connection failed.")
   else:
       st.warning("Start a new session and upload a document to begin.")

st.header("Chat with Document")
if not st.session_state.processed_filename:
   st.info("Your chat session will appear here once a document is processed.")

for message in st.session_state.messages:
   with st.chat_message(message["role"]):
       st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the document..."):
   if st.session_state.processed_filename:
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"): st.markdown(prompt)
       with st.chat_message("assistant"):
           with st.spinner("Thinking..."):
               try:
                   response = requests.post(f"{ORCHESTRATOR_URL}/chat/", json={"query": prompt})
                   if response.status_code == 200:
                       answer = response.json().get("answer", "No answer found.")
                       st.markdown(answer)
                       st.session_state.messages.append({"role": "assistant", "content": answer})
                   else:
                       st.error(f"Error: {response.status_code} - {response.text}")
               except requests.ConnectionError:
                   st.error("Connection failed.")
   else:
       st.warning("Please upload a document before asking questions.")
