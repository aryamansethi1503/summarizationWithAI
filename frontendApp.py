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
        full_text = [para.text for para in doc.paragraphs]
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
st.title("📄 DocQuery AI: Multi-Document Q&A and Summarization")

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "upload_key" not in st.session_state:
    st.session_state.upload_key = 0
if "last_summary" not in st.session_state:
    st.session_state.last_summary = None

with st.sidebar:
    st.header("Start New Session")
    if st.button("Clear Database and Start New", use_container_width=True):
        with st.spinner("Starting new session..."):
            try:
                requests.post(f"{ORCHESTRATOR_URL}/new-session/")
                st.session_state.processed_files = []
                st.session_state.messages = []
                st.session_state.upload_key += 1
                st.session_state.last_summary = None
                st.success("Ready for new documents.")
                st.rerun()
            except requests.ConnectionError:
                st.error("Connection failed. Is the orchestrator running?")

    st.divider()
    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more files",
        type=['pdf', 'png', 'jpg', 'jpeg', 'txt', 'docx'],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.upload_key}"
    )

    if uploaded_files:
        new_files_to_process = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
        if new_files_to_process:
            st.session_state.last_summary = None
            with st.spinner(f"Processing {len(new_files_to_process)} new file(s)..."):
                for uploaded_file in new_files_to_process:
                    st.info(f"Processing {uploaded_file.name}...")
                    text = None
                    file_ext = os.path.splitext(uploaded_file.name)[-1].lower()
                    if file_ext == ".pdf": text = extract_text_from_pdf(uploaded_file)
                    elif file_ext in [".png", ".jpg", ".jpeg"]: text = extract_text_from_image(uploaded_file)
                    elif file_ext == ".txt": text = extract_text_from_txt(uploaded_file)
                    elif file_ext == ".docx": text = extract_text_from_docx(uploaded_file)
                    
                    if text:
                        chunks = [text[i:i + CHUNK_SIZE_CHARS] for i in range(0, len(text), CHUNK_SIZE_CHARS)]
                        for i, chunk in enumerate(chunks):
                            payload = {"chunk": chunk, "filename": uploaded_file.name, "chunk_index": i}
                            requests.post(f"{ORCHESTRATOR_URL}/upload-chunk/", json=payload)
                        st.session_state.processed_files.append(uploaded_file.name)
                    else:
                        st.error(f"Could not extract text from {uploaded_file.name}.")
            st.success("All new files processed!")
            st.rerun()

    st.divider()
    st.header("Actions")
    if st.session_state.processed_files:
        if st.button("Generate Synthesis Summary", use_container_width=True, type="primary"):
            with st.spinner("Generating synthesis summary from all documents..."):
                try:
                    response = requests.post(f"{ORCHESTRATOR_URL}/summarize-all/")
                    if response.status_code == 200:
                        summary = response.json().get("summary")
                        st.session_state.last_summary = summary
                        st.session_state.messages.insert(0, {"role": "assistant", "content": f"**Synthesis Summary:**\n\n{summary}"})
                        st.rerun()
                    else:
                        st.error(f"Error during summarization: {response.status_code} - {response.text}")
                except requests.ConnectionError:
                    st.error("Connection failed.")
    else:
        st.warning("Upload documents to begin.")

tab1, tab2 = st.tabs(["💬 Chat & Results", "🗂️ Processed Files"])

with tab1:
    if not st.session_state.processed_files:
        st.info("Your chat session will appear here once documents are processed.")
    else:
        st.info(f"Chatting with knowledge from {len(st.session_state.processed_files)} document(s).")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.last_summary:
        st.divider()
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Translate Summary to Hindi", use_container_width=True):
                with st.spinner("Translating to Hindi..."):
                    try:
                        payload = {"text": st.session_state.last_summary, "language": "Hindi"}
                        response = requests.post(f"{ORCHESTRATOR_URL}/translate/", json=payload)
                        if response.status_code == 200:
                            translated_summary = response.json().get("translated_text")
                            st.session_state.messages.append({"role": "assistant", "content": f"**Hindi Translation:**\n\n{translated_summary}"})
                            st.session_state.last_summary = None
                            st.rerun()
                        else:
                            st.error(f"Error during translation: {response.status_code} - {response.text}")
                    except requests.ConnectionError:
                        st.error("Connection failed during translation.")
    
    prompt_placeholder = "Ask a question or type /challenge <your statement>"
    if prompt := st.chat_input(prompt_placeholder):
        if st.session_state.processed_files:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    try:
                        if prompt.lower().startswith("/challenge "):
                            statement = prompt[len("/challenge "):].strip()
                            endpoint = "/challenge/"
                            payload = {"statement": statement}
                        else:
                            endpoint = "/chat/"
                            payload = {"query": prompt}

                        response = requests.post(f"{ORCHESTRATOR_URL}{endpoint}", json=payload)
                        
                        if response.status_code == 200:
                            answer = response.json().get("answer", "No answer found.")
                            st.markdown(answer)
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    except requests.ConnectionError:
                        st.error("Connection failed.")
        else:
            st.warning("Please upload documents before asking questions.")

with tab2:
    st.header("Manage Processed Documents")
    if st.session_state.processed_files:
        for name in st.session_state.processed_files[:]:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"- {name}")
            with col2:
                if st.button("Remove", key=f"remove_{name}", use_container_width=True):
                    with st.spinner(f"Removing {name}..."):
                        try:
                            response = requests.post(f"{ORCHESTRATOR_URL}/delete-document/", json={"filename": name})
                            if response.status_code == 200:
                                st.session_state.processed_files.remove(name)
                                st.success(f"Removed {name}.")
                                st.rerun()
                            else:
                                st.error(f"Failed to remove {name}.")
                        except requests.ConnectionError:
                            st.error("Connection failed.")
    else:
        st.info("No documents have been processed in this session.")