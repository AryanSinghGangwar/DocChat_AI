# 📄 DocChat AI: Full-Stack RAG Application

An intelligent document interaction pipeline that allows users to upload PDF documents and ask context-aware questions. It uses **Retrieval-Augmented Generation (RAG)** powered by Hugging Face Sentence Transformers and Google's Gemini 2.0 API. 

This repository is structured as a modern **microservices architecture** using FastAPI (Backend), Streamlit (Frontend), and Docker.

---

## 🧠 Interview Revision Guide: Core Concepts

This section is designed to help you revise the underlying AI and software engineering concepts used to build this project.

### 1. Retrieval-Augmented Generation (RAG)
**What it is:** LLMs like Gemini have a knowledge cutoff and don't know your private PDF data. RAG solves this by *retrieving* relevant information from your document first, and then *augmenting* the LLM's prompt with that information before asking it to *generate* an answer.
**Why we used it:** It prevents hallucinations and allows the model to ground its answers in the exact text of the uploaded PDF.

### 2. Text Embeddings
**What it is:** We convert chunks of text into lists of numbers (dense vectors). Words with similar meanings will have vectors that are closer together in a high-dimensional space.
**Implementation:** We use the `all-MiniLM-L6-v2` model via the `sentence-transformers` library. It's a lightweight, fast, and highly effective model for mapping sentences to a 384-dimensional vector space.

### 3. Cosine Similarity (Vector Math)
**What it is:** To find which part of the PDF answers the user's question, we convert the user's question into an embedding vector. We then calculate the Cosine Similarity between the question's vector and the vectors of all the PDF pages. 
**Implementation:** We use `util.pytorch_cos_sim()` to find the highest overlapping vectors (the most relevant text chunks). Cosine similarity measures the angle between two vectors, ranging from -1 to 1 (1 being identical).

### 4. Microservice Architecture
**What it is:** Instead of a single monolithic script, the system is decoupled into two services:
*   **Backend (FastAPI):** Handles heavy lifting (ML models, parsing, APIs). Highly scalable.
*   **Frontend (Streamlit):** Handles UI and state. Can be swapped out for Next.js/React later without rewriting the AI logic.

---

## ⚙️ Step-by-Step Implementation Under the Hood

If asked how you built this from scratch, here is the exact flow of data through the system:

### Phase 1: Ingestion & Upload
1. The user uploads a `.pdf` file via the **Streamlit UI**.
2. Streamlit sends a `POST /upload` request (using `multipart/form-data`) to the **FastAPI Backend**.
3. FastAPI saves the file locally in an `uploads/` directory to simulate cloud blob storage (like AWS S3).

### Phase 2: Processing & Retrieval
1. The user types a question in the UI. Streamlit sends a `POST /chat` request containing the `document_id` and the `query`.
2. **Text Extraction:** FastAPI triggers the pipeline. `pdfplumber` opens the PDF and extracts raw text page by page.
3. **Embedding Generation:** The `SentenceTransformer` model processes the extracted text, converting every page's text into mathematical vectors (embeddings).
4. **Semantic Matching:** The user's query is also converted into a vector. PyTorch calculates the Cosine Similarity between the query vector and all page vectors to find the `top_k` most relevant pages.

### Phase 3: Generation
1. **Prompt Construction:** The backend constructs a prompt combining the user's original question with the raw text retrieved from the most similar PDF page.
2. **LLM Inference:** An HTTP POST request is sent to Google's Gemini 2.0 API with the constructed prompt.
3. **Response:** Gemini generates a human-readable explanation based *only* on the provided context. FastAPI returns this string back to Streamlit, which displays it in the chat UI.

---

## 🏗️ Architecture & Pipeline Diagram

```mermaid
flowchart TD
    User[User (Browser)] -->|Uploads PDF & Types Question| UI[Streamlit Frontend]
    UI -->|REST API Calls| API[FastAPI Backend]
    
    subgraph Backend [FastAPI Application (Port 8000)]
        API -->|Extract| Parse[PDFPlumber]
        Parse -->|Raw Text| Embed[Sentence Transformers]
        Embed -->|Vector Match| Math[PyTorch Cosine Sim]
        Math -->|Top Context| Builder[Prompt Builder]
    end
    
    Builder -->|Context + Query| External[Google Gemini API]
    External -->|Generated Answer| API
    API -->|JSON Response| UI
```

---

## 🚀 Setup & Installation Instructions

### Prerequisites
- Docker and Docker Compose
- A Gemini API Key from Google AI Studio.

### Option 1: Quick Start (Docker) - Recommended

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AryanSinghGangwar/DocChat_AI.git
   cd DocChat_AI
   ```

2. **Configure Environment:**
   ```bash
   cp .env.example .env
   # Open .env and add your GEMINI_API_KEY
   ```

3. **Spin up the microservices:**
   ```bash
   docker-compose up --build
   ```

4. **Access the Application:**
   - **Chat Interface:** [http://localhost:8501](http://localhost:8501)
   - **API Documentation (Swagger):** [http://localhost:8000/docs](http://localhost:8000/docs)

### Option 2: Manual Setup (Local Python)

**Terminal 1 (Backend):**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
cd frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Enhancements (To discuss in interviews)
If an interviewer asks "How would you scale this?", mention these:
1. **Vector Database:** Move away from in-memory PyTorch matching to a dedicated DB like ChromaDB or Pinecone for scaling to millions of documents.
2. **Advanced Chunking:** Instead of chunking by "page", use semantic chunking or LangChain's `RecursiveCharacterTextSplitter` with overlaps to ensure context isn't cut off mid-sentence.
3. **Conversational Memory:** Store past user questions in Redis or SQLite so the LLM remembers previous turns in the chat (Memory Windowing).
