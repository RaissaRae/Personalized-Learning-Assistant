# Personalized-Learning-Assistant

## Overview
The **Personalized Learning Assistant** is an AI-driven educational tool that helps users engage with course materials more effectively. It provides tailored summaries, detailed explanations, and quizzes based on uploaded course documents. The application is built using **Streamlit** and employs **RAG (Retrieval-Augmented Generation)**, a **vector database**, and **Groq API (default LLM)** to deliver relevant and personalized learning experiences.

## Features
- **Streamlit User Interface**  
  - Home page
  - Upload section for additional materials
  - Dropdown to select the LLM (default: Groq API)
  - Chatbot interface with references to source materials

- **Content Ingestion Agent**  
  - Processes and stores reference materials (PDFs) in a vector database

- **Question Answering Agent**  
  - Uses RAG to retrieve relevant information and provide accurate responses

### Possible Extensions
- **Additional Agents:**
  - **CheatSheet Agent**: Generates one-page summaries
  - **Quiz Agent**: Creates multiple-choice or open-ended questions

- **Multiple LLM Support:**
  - Option to select different LLMs (OpenAI, Hugging Face, etc.)
  - Stores embeddings based on the chosen LLM
  - Hugging Face API or running models locally using **ollama**

- **Alternative Vector Databases:**
  - FAISS

- **Expanded Content Ingestion:**
  - Extracts information from **YouTube videos**
  - Parses PowerPoint slides, including graphs and images (**Docling integration**)
  - Incorporates web search functionality

- **Memory Enhancements:**
  - **Short-term memory** for referencing earlier topics in a chat session
  - **Long-term memory** for personalized responses based on previous interactions

- **Voice-enabled Chat Interface:**
  - **Speech-to-Text (STT):** Converts voice input into text for processing
    - Supports automatic endpointing for seamless interaction
  - **Text-to-Speech (TTS):** Converts LLM-generated responses into audio
    - Supports streaming for real-time voice output

## Usage
1. Open the Streamlit app in your browser.
2. Upload your course documents (PDFs, slides, etc.).
3. Select your preferred LLM (default: Groq API).
4. Use the chatbot to ask questions, get summaries, or generate quizzes.
5. (Optional) Enable voice interactions for a hands-free experience.
