import streamlit as st
from dotenv import load_dotenv
import os
from groq import Groq
import fitz  # PyMuPDF for handling PDFs
from docx import Document  # python-docx for handling Word documents
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Any
import requests
import openai
import google.generativeai as genai
import numpy as np
import faiss
import pickle
from datetime import datetime
import sqlite3
from chat_memory import ChatMemorySystem 
import uuid


import requests
import json
from deepgram import Deepgram
import wave
import pyaudio
import asyncio
import io
import boto3


# Import the content ingestion system
from content_injestion import ContentIngestionSystem, render_content_ingestion_ui

# Initialize the content ingestion system with your Gemini API key
ingestion_system = ContentIngestionSystem(gemini_api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize memory system
memory_system = ChatMemorySystem(memory_path="./chat_memory")
# Generate a session ID for each user session
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# In your chat interface
def handle_user_input(user_input: str):
    # Generate response using your existing function
    response = generate_answer(user_input, context=context_str, llm_provider=st.session_state.get('llm_provider', 'groq'))
    
    # Store in memory
    memory_system.add_message_to_short_term("user", user_input)
    memory_system.add_message_to_short_term("assistant", response)
    memory_system.add_to_long_term(
        content=user_input,
        role="user",
        session_id=st.session_state.session_id
    )
    memory_system.add_to_long_term(
        content=response,
        role="assistant",
        session_id=st.session_state.session_id
    )
    
    return response

# Load environment variables from the .env file
env_path = r"Rae\.env"
load_dotenv(dotenv_path=env_path)

# Initialize the Groq client
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("Error: GROQ_API_KEY is not defined in the environment variables.")
    st.stop()

client = Groq(api_key=groq_api_key)

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY:
    st.error("Error: DEEPGRAM_API_KEY not found in the environment variables.")
    st.stop() 

# Initialize the Polly client
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_region = os.getenv("AWS_REGION", "us-east-1")  # Default to 'us-east-1' if region is not in .env

polly_client = boto3.client(
    "polly",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# Initialize Deepgram client
deepgram = Deepgram(DEEPGRAM_API_KEY)


#function to convert text to speech
def synthesize_speech(text: str, voice_id: str = "Joanna") -> bytes:
    try:
        response = polly_client.synthesize_speech(
            Text=text,
            OutputFormat="mp3",
            VoiceId=voice_id
        )

        # Return the audio stream as bytes
        if "AudioStream" in response:
            return response["AudioStream"].read()
        else:
            st.error("Could not retrieve audio stream from Polly.")
            return b""
    except Exception as e:
        st.error(f"An error occurred while generating speech: {str(e)}")
        return b""

# Function to record and transcribe audio using Deepgram
async def record_and_transcribe():
    # Record audio from the microphone using PyAudio
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)
    frames = []

    st.write("Recording... Speak now!")
    for i in range(0, int(16000 / 1024 * 5)):  # Record 5 seconds
        data = stream.read(1024)
        frames.append(data)

    st.write("Recording finished.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file in memory (not disk)
    audio_buffer = io.BytesIO()
    wf = wave.open(audio_buffer, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

    audio_buffer.seek(0)

        # Prepare the audio data in the correct format
    audio_data = {
        'buffer': audio_buffer,  # Send the audio buffer wrapped in a dictionary
        'mimetype': 'audio/wav'  # Specify the mimetype of the file
    }

        # Send the audio buffer to Deepgram API for transcription (as binary)
    response = await deepgram.transcription.prerecorded(
        audio_data,  # Read the entire buffer as bytes
        {
            'language': 'en',
            'punctuate': True
        }
    )
    
    # Extract transcription from the response
    transcription = response['results']['channels'][0]['alternatives'][0]['transcript']
    return transcription



# Initialize OpenAI client
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    openai.api_key = openai_api_key

# Initialize Google Gemini client
gemini_api_key = os.getenv("GOOGLE_API_KEY")
if gemini_api_key:
    genai.configure(api_key=gemini_api_key)

huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")



class VectorStore:
    def __init__(self, dimension: int = 384, index_directory: str = "./vector_store"):
        """Initialize FAISS vector store with SQLite metadata storage"""
        self.dimension = dimension
        self.index_directory = index_directory
        os.makedirs(index_directory, exist_ok=True)
        
        self.index_path = os.path.join(index_directory, "faiss.index")
        self.db_path = os.path.join(index_directory, "metadata.db")
        
        # Initialize FAISS index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatL2(dimension)
        
        # Initialize SQLite
        self.init_sqlite()
        
        # Initialize embedding model
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def init_sqlite(self):
        """Initialize SQLite database schema"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vector_id INTEGER UNIQUE,
                    document TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP
                )
            """)
            conn.commit()

    def add(self, documents: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store"""
        if not documents:
            return
            
        if metadatas is None:
            metadatas = [{"source": "uploaded"} for _ in documents]
        
        # Get embeddings
        embeddings = self.embedding_function.embed_documents(documents)
        
        # Add to FAISS
        vectors = np.array(embeddings, dtype=np.float32)
        vector_ids = np.arange(self.index.ntotal, self.index.ntotal + len(vectors))
        self.index.add(vectors)
        
        # Save to SQLite
        with sqlite3.connect(self.db_path) as conn:
            for vector_id, document, metadata in zip(vector_ids, documents, metadatas):
                conn.execute(
                    "INSERT INTO documents (vector_id, document, metadata, created_at) VALUES (?, ?, ?, ?)",
                    (int(vector_id), document, str(metadata), datetime.now())
                )
            conn.commit()
        
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

    def query(self, query_texts: List[str], n_results: int = 5) -> Dict[str, Any]:
        """Query the vector store for similar documents"""
        query_embeddings = self.embedding_function.embed_documents(query_texts)
        query_vectors = np.array(query_embeddings, dtype=np.float32)
        
        # Search FAISS
        distances, indices = self.index.search(query_vectors, n_results)
        
        results = {
            "documents": [],
            "metadatas": [],
            "distances": distances.tolist()
        }
        
        # Retrieve from SQLite
        with sqlite3.connect(self.db_path) as conn:
            for query_indices in indices:
                query_docs = []
                query_metas = []
                for idx in query_indices:
                    row = conn.execute(
                        "SELECT document, metadata FROM documents WHERE vector_id = ?",
                        (int(idx),)
                    ).fetchone()
                    if row:
                        query_docs.append(row[0])
                        query_metas.append(eval(row[1]))
                results["documents"].append(query_docs)
                results["metadatas"].append(query_metas)
        
        return results

# Initialize vector store
vector_store = VectorStore(dimension=384, index_directory="./vector_store")

# Function to add text chunks to vector store
def add_to_vector_store(text: str):
    """Add text chunks to the vector store"""
    if text.strip() == "":
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(text)
    
    vector_store.add(
        documents=text_chunks,
        metadatas=[{"source": "uploaded"} for _ in text_chunks]
    )

# Function to retrieve context from vector store
def retrieve_from_vector_store(query: str) -> Dict[str, Any]:
    """Retrieve similar documents from vector store"""
    try:
        results = vector_store.query(
            query_texts=[query],
            n_results=5
        )
        return {
            "documents": results["documents"][0],  # First query results
            "metadata": results["metadatas"][0]
        }
    except Exception as e:
        return {
            "documents": [],
            "metadata": [],
            "error": f"An error occurred during vector store retrieval: {str(e)}"
        }

   
def search_google(query: str) -> list:
    api_key = "xx"  
    cse_id = "a278fa989c26c4d0f"  
    endpoint = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": api_key,
        "cx": cse_id,
        "num": 10 # Get the top 3 results
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        return [
            {"title": item["title"], "url": item["link"]}
            for item in data.get("items", [])
        ]
    except Exception as e:
        return [{"title": "Error", "url": str(e)}]
    
# Function to summarize text
def summarize_text(context: str) -> str:
    try:
    # Use vector store instead of ChromaDB
        vector_store_results = retrieve_from_vector_store("Summarize this content")
        documents = vector_store_results.get("documents", [])
        metadata = vector_store_results.get("metadata", [])

        # Rest of your summarize_text function remains the same
        retrieved_context = "\n".join(documents)
        full_context = context + "\n" + retrieved_context

        # Generate the summary
        prompt = f"Summarize the following content:\n{full_context}\nSummary:"
        summary = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an assistant that summarizes content and provides references."},
                {"role": "user", "content": prompt},
            ],
            model="llama3-8b-8192",
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            stream=False,
        )
        summary_text = summary.choices[0].message.content.strip()

        # Add document references
        references = "\n\nReferences:\n"
        for meta in metadata:
            if isinstance(meta, dict) and "source" in meta:
                references += f"- Source: {meta['source']}\n"

        return summary_text + references

    except Exception as e:
        return f"An error occurred during summarization: {str(e)}"
    
def extract_keywords(text: str) -> str:
    """Extracts keywords from the assistant's response."""
    # Split text into sentences or meaningful chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=150, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # Take the most relevant parts (e.g., first two sentences)
    keywords = " ".join(chunks[:2])  # Adjust number of chunks as needed
    return keywords
# Function to generate a response using the Groq SDK
def generate_answer(query: str, context: str = "", llm_provider: str = "groq") -> str:
    try:
    # Use vector store instead of ChromaDB
        vector_store_results = retrieve_from_vector_store(query)
        documents = vector_store_results.get("documents", [])
        metadata = vector_store_results.get("metadata", [])

        # Rest of your generate_answer function remains the same
        retrieved_context = "\n".join(documents)
        full_context = context + "\n" + retrieved_context

        # Step 2: Generate the Assistant's Answer based on the selected LLM
        if llm_provider == "groq":
            # Existing Groq implementation
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that always provides accurate responses."},
                    {"role": "user", "content": full_context + "\nQuestion: " + query + "\nAnswer:"},
                ],
                model="llama3-8b-8192",
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            assistant_answer = chat_completion.choices[0].message.content.strip()
        
        elif llm_provider == "openai":
            # OpenAI GPT implementation
            chat_completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that always provides accurate responses."},
                    {"role": "user", "content": full_context + "\nQuestion: " + query + "\nAnswer:"},
                ],
                temperature=0.5,
                max_tokens=1024
            )
            assistant_answer = chat_completion.choices[0].message.content.strip()
        
        elif llm_provider == "gemini":
            # Google Gemini implementation
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                full_context + "\nQuestion: " + query + "\nAnswer:",
                generation_config={
                    "temperature": 0.5,
                    "max_output_tokens": 1024
                }
            )
            assistant_answer = response.text.strip()
        
        elif llm_provider == "huggingface":
            # Hugging Face API implementation
            huggingface_api_url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
            headers = {
                "Authorization": f"Bearer {huggingface_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": f"System: You are a helpful assistant that always provides accurate responses.\n\nContext: {full_context}\n\nQuestion: {query}\n\nAnswer:",
                "parameters": {
                    "max_new_tokens": 1024,
                    "temperature": 0.5,
                    "top_p": 1.0
                }
            }
            
            response = requests.post(huggingface_api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                # Hugging Face returns a list of generations
                assistant_answer = response.json()[0]['generated_text'].split("Answer:")[-1].strip()
            else:
                raise Exception(f"Hugging Face API error: {response.text}")
        
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

        # Step 3: Extract Keywords from the Assistant's Answer
        keywords = extract_keywords(assistant_answer)

        # Step 4: Use Extracted Keywords for Web Search
        web_references = search_google(keywords)

        # Step 5: Format Web References
        web_references_section = "\n\nWeb References:\n"
        for ref in web_references:
            web_references_section += f"- [{ref['title']}]({ref['url']})\n"

        # Combine the answer with web references
        return assistant_answer + web_references_section

    except Exception as e:
        return f"An error occurred: {str(e)}"
    
def find_related_sources(context: str) -> str:
    try:
        # Extract key phrases or summary from the document
        keywords = " ".join(context.split()[:10])  # First 10 words as keywords (you can refine this)

        # Perform web search using the extracted keywords
        web_references = search_google(keywords)

        # Format web references
        web_references_section = "\n\nWeb References:\n"
        for ref in web_references:
            web_references_section += f"- [{ref['title']}]({ref['url']})\n"

        return web_references_section

    except Exception as e:
        return f"An error occurred while finding related sources: {str(e)}"

# Streamlit setup
st.set_page_config(page_title="Personalized Learning Assistant", layout="wide")
st.title("📘 Personalized Learning Assistant")

st.sidebar.header("Upload Your Documents Here 📄")
uploaded_files = st.sidebar.file_uploader("Upload your PDFs or Word documents:", type=["pdf", "docx"], accept_multiple_files=True)

uploaded_text = ""
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            try:
                doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                for page in doc:
                    text_content = page.get_text()
                    uploaded_text += text_content
                    add_to_vector_store(text_content)  # Use new vector store function
            except Exception as e:
                st.sidebar.error(f"Error reading PDF: {str(e)}")
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = Document(uploaded_file)
                for para in doc.paragraphs:
                    text_content = para.text
                    uploaded_text += text_content + "\n"
                    add_to_vector_store(text_content)  # Use new vector store function
            except Exception as e:
                st.sidebar.error(f"Error reading Word: {str(e)}")

tabs = st.tabs(["Ask a Question", "Summary Agent", "Quiz Agent", "Citation Finder Agent", "Related Sources Agent", "Content Ingestion"])

def handle_audio_transcription(tab_name: str):
    # Dynamically generate a unique key for each button based on the tab name
    button_key = f"record_and_transcribe_audio_button_{tab_name}"
    
    # Ensure each button has a unique key
    if st.button("Record and Transcribe Audio", key=button_key):
        #transcription = record_and_transcribe()
        transcription = asyncio.run(record_and_transcribe())  # Use asyncio.run() to run the async function
        st.session_state.transcription = transcription  # Store transcription in session_state
        #st.write(f"**Transcription:** {transcription}")

        # Update the text input with the transcription
        #query = st.text_input("Enter your question here:", value=transcription)
        return transcription
    return None


# Add testing section in sidebar (before the tabs)
st.sidebar.header("Memory Testing")
if st.sidebar.button("Test Memory"):
    st.sidebar.write("Testing memory system...")
    
    # Test short-term memory
    st.sidebar.subheader("Short-term Memory")
    if memory_system.short_term_messages:
        st.sidebar.write(f"Number of messages in short-term memory: {len(memory_system.short_term_messages)}")
        st.sidebar.write("Recent messages:")
        for msg in memory_system.short_term_messages[-3:]:  # Show last 3 messages
            st.sidebar.write(f"{msg['role']}: {msg['content'][:50]}...")
    else:
        st.sidebar.write("No messages in short-term memory")
    
    # Test long-term memory
    st.sidebar.subheader("Long-term Memory")
    db_path = os.path.join("./chat_memory", "long_term_memory.db")
    if os.path.exists(db_path):
        with sqlite3.connect(db_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
            recent = conn.execute(
                "SELECT timestamp, role, content FROM conversations ORDER BY timestamp DESC LIMIT 3"
            ).fetchall()
        st.sidebar.write(f"Total conversations stored: {count}")
        st.sidebar.write("Recent stored conversations:")
        for timestamp, role, content in recent:
            st.sidebar.write(f"{timestamp} - {role}: {content[:50]}...")
    else:
        st.sidebar.write("No long-term memory database found")



with tabs[0]:
    st.header("Ask Your Question 🤔")

    # Create columns for the header area
    col1, col2, col3, col4 = st.columns([1, 3, 1, 0.5])
    
    # LLM provider selection
    provider_map = {
        "Groq (Llama3)": "groq",
        "OpenAI (GPT-3.5)": "openai", 
        "Google Gemini": "gemini",
        "Hugging Face (Llama-2)": "huggingface"
    }
    
    with col1:
        st.write("AI Model:")
    with col2:
        llm_provider = st.selectbox(
            "",
            list(provider_map.keys()),
            key="llm_provider",
            label_visibility="collapsed"
        )
    
    # Clear chat button
    with col4:
        if st.button("🗑️", help="Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            memory_system.setup_short_term_memory()
            st.rerun()

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Handle transcription and add it to chat input
    transcription = handle_audio_transcription("tab_1")
    if transcription:
        # Add the transcription as a user message
        st.session_state.chat_history.append({"role": "user", "content": transcription})

        # Automatically generate a response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                context_str = memory_system.format_context(transcription)
                response = generate_answer(
                    transcription,
                    context=context_str,
                    llm_provider=provider_map[llm_provider]
                )
                st.write(response)

                # Strip web references for TTS (before displaying to user)
                if "Web References:" in response:
                    main_text, _ = response.split("Web References:", 1)
                    response_for_tts = main_text
                else:
                    response_for_tts = response

                # Generate TTS for the assistant's response (before showing the web references)
                audio_data = synthesize_speech(response_for_tts)  # Call the TTS function
                if audio_data:
                    st.audio(audio_data, format="audio/mp3", start_time=0)  # Play the audio

        # Append assistant response to chat history (with web references)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Update memory systems
        memory_system.add_message_to_short_term("user", transcription)
        memory_system.add_message_to_short_term("assistant", response)
        memory_system.add_to_long_term(
            content=transcription,
            role="user",
            session_id=st.session_state.session_id
        )
        memory_system.add_to_long_term(
            content=response,
            role="assistant",
            session_id=st.session_state.session_id
        )
        st.rerun()

    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if i > 0:
                st.divider()
            if message["role"] == "user":
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    content = message["content"]

                    # Strip web references before sending it to TTS (before display)
                    if "Web References:" in content:
                        main_text, refs = content.split("Web References:", 1)
                        st.markdown(main_text)
                        st.markdown("---")
                        st.markdown("**📚 Web References:**" + refs)

                        content_for_tts = main_text  # Only pass the main content for TTS
                    else:
                        st.markdown(content)
                        content_for_tts = content  # No need to strip anything if no web references

                    # Generate TTS for the assistant responses (before displaying web references)
                    audio_data = synthesize_speech(content_for_tts)  # Call the TTS function
                    if audio_data:
                        st.audio(audio_data, format="audio/mp3", start_time=0)  # Play the audio

    # Chat input at the bottom
    prompt = st.chat_input("What would you like to know?")
    if prompt:
        # Add user input to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                context_str = memory_system.format_context(prompt)
                response = generate_answer(
                    prompt,
                    context=context_str,
                    llm_provider=provider_map[llm_provider]
                )
                st.write(response)

                # Strip web references for TTS (before displaying to user)
                if "Web References:" in response:
                    main_text, _ = response.split("Web References:", 1)
                    response_for_tts = main_text
                else:
                    response_for_tts = response

                # Generate TTS for the assistant's response (before showing the web references)
                audio_data = synthesize_speech(response_for_tts)  # Call the TTS function
                if audio_data:
                    st.audio(audio_data, format="audio/mp3", start_time=0)  # Play the audio

        # Append response to chat history (with web references)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Update memory systems
        memory_system.add_message_to_short_term("user", prompt)
        memory_system.add_message_to_short_term("assistant", response)
        memory_system.add_to_long_term(
            content=prompt,
            role="user",
            session_id=st.session_state.session_id
        )
        memory_system.add_to_long_term(
            content=response,
            role="assistant",
            session_id=st.session_state.session_id
        )

        
# "Summary Agent" tab
with tabs[1]:
    st.header("Summary Agent 📝")
    if st.button("Generate Summary", key="summary_agent"):
        if uploaded_text.strip():
            summary = summarize_text(uploaded_text)
            st.write(f"**Summary:**\n{summary}")

            # Exclude references from the audio
            clean_summary = summary.split("References:")[0].strip()  # Remove everything after "References:"
            
            # Generate and play audio for the clean summary
            audio_data = synthesize_speech(clean_summary)  # Use the cleaned summary as input for TTS
            if audio_data:
                st.audio(audio_data, format="audio/mp3", start_time=0)
            
        else:
            st.error("No uploaded content available for summarization.")


# Function to generate a quiz
def generate_quiz(context: str, llm_provider: str = "groq") -> str:
    try:
        if llm_provider == "groq":
            prompt = f"Create a quiz with multiple-choice and fill-in-the-blank questions based on the following content:\n{context}\nQuiz:"
            quiz_response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that generates quiz questions."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False,
            )
            return quiz_response.choices[0].message.content
        elif llm_provider == "gemini":
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                f"Create a quiz with multiple-choice and fill-in-the-blank questions based on the following content:\n{context}\nQuiz:",
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 1024
                }
            )
            return response.text
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    except Exception as e:
        return f"An error occurred while generating the quiz: {str(e)}"

# "Quiz Agent" tab
with tabs[2]:
    st.header("Quiz Agent 🎓")
    
    # Add LLM provider selection
    llm_provider = st.selectbox(
        "Select LLM Provider:",
        ["Groq (Llama3)", "Google Gemini"],
        key="quiz_llm_provider"
    )
    
    # Map selection to provider string
    provider_map = {
        "Groq (Llama3)": "groq",
        "Google Gemini": "gemini"
    }
    
    if st.button("Generate Quiz", key="quiz_agent"):
        if uploaded_text.strip():
            selected_provider = provider_map[llm_provider]
            quiz = generate_quiz(uploaded_text, llm_provider=selected_provider)
            st.write(f"**Quiz:**\n{quiz}")
        else:
            st.error("No uploaded content available to generate a quiz.")
    


# Function to find citations
def find_citations(context: str, llm_provider: str = "groq") -> str:
    try:
        if llm_provider == "groq":
            prompt = f"Extract all citations, references, and URLs from the following content:\n{context}\nCitations:"
            citation_response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that extracts citations and references from content."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=512,
                top_p=1,
                stream=False,
            )
            return citation_response.choices[0].message.content
        elif llm_provider == "gemini":
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                f"Extract all citations, references, and URLs from the following content:\n{context}\nCitations:",
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 512
                }
            )
            return response.text
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    except Exception as e:
        return f"An error occurred while finding citations: {str(e)}"

# "Citation Finder Agent" tab
with tabs[3]:
    st.header("Citation Finder Agent 🔍")    
    if st.button("Find Citations", key="citation_finder_agent"):
        if uploaded_text.strip():
            citations = generate_answer(uploaded_text)
            st.write(f"**Citations Found:**\n{citations}")
        else:
            st.error("No uploaded content available.")

    
    # Add LLM provider selection
    llm_provider = st.selectbox(
        "Select LLM Provider:",
        ["Groq (Llama3)", "Google Gemini"],
        key="citation_llm_provider"
    )
    
    # Map selection to provider string
    provider_map = {
        "Groq (Llama3)": "groq",
        "Google Gemini": "gemini"
    }
    

# Function to find related sources
def find_related_sources(context: str, llm_provider: str = "groq") -> str:
    try:
        # Extract key phrases or summary from the document
        keywords = " ".join(context.split()[:10])  # First 10 words as keywords (you can refine this)

        if llm_provider == "groq":
            prompt = f"Perform a web search using the following keywords and provide a list of related sources:\nKeywords: {keywords}\nWeb References:"
            search_response = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are an assistant that finds related sources based on given keywords."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama3-8b-8192",
                temperature=0.5,
                max_tokens=512,
                top_p=1,
                stream=False,
            )
            return search_response.choices[0].message.content
        elif llm_provider == "gemini":
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(
                f"Perform a web search using the following keywords and provide a list of related sources:\nKeywords: {keywords}\nWeb References:",
                generation_config={
                    "temperature": 0.5,
                    "max_output_tokens": 512
                }
            )
            return response.text
        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    except Exception as e:
        return f"An error occurred while finding related sources: {str(e)}"

# "Find Related Sources" tab
with tabs[4]:
    st.header("Find Related Sources 🌐")
    
    # Add LLM provider selection
    llm_provider = st.selectbox(
        "Select LLM Provider:",
        ["Groq (Llama3)", "Google Gemini"],
        key="related_sources_llm_provider"
    )
    
    # Map selection to provider string
    provider_map = {
        "Groq (Llama3)": "groq",
        "Google Gemini": "gemini"
    }
    
    if st.button("Find Related Sources", key="related_sources_agent"):
        if uploaded_text.strip():
            selected_provider = provider_map[llm_provider]
            related_sources = find_related_sources(uploaded_text, llm_provider=selected_provider)
            st.write(f"**Related Sources Found:**\n{related_sources}")
        else:
            st.error("No uploaded content available to find related sources.")


with tabs[5]:
    st.header("Content Ingestion 📥")

    # Call the function to render the UI for content ingestion
    render_content_ingestion_ui(ingestion_system)

    # Optional Text Area to Display Ingested Content
    ingested_content = st.session_state.get("ingested_content", "")
    if ingested_content.strip():
        st.markdown("### Preview of Ingested Content:")
        st.text_area("Ingested Content", ingested_content, height=200)

        # Add TTS Button to Read the Ingested Content
        if st.button("🔊 Read Ingested Content", key="read_ingested_content"):
            if ingested_content:
                st.info("Reading the ingested content aloud...")
                audio_data = synthesize_speech(ingested_content)  # TTS function
                if audio_data:
                    st.audio(audio_data, format="audio/mp3", start_time=0)
            else:
                st.warning("No content available to read aloud.")