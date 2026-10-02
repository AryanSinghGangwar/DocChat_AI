# DocChat AI

DocChat AI is an intelligent document interaction pipeline that allows users to ask questions about their PDF documents and receive accurate, context-aware answers powered by the Gemini 2.0 API. 

**Now upgraded to a production-ready microservice architecture with a FastAPI backend and a Streamlit frontend.**

## Features

- **Automated Text Extraction**: Efficiently extracts readable text from PDF documents using `pdfplumber`.
- **Semantic Search**: Utilizes `sentence-transformers` (all-MiniLM-L6-v2) to find the most relevant sections of the document based on the user's query.
- **LLM-Powered Explanations**: Integrates with the Gemini API to synthesize and explain the extracted context clearly and thoroughly.
- **Modern Microservice Architecture**: 
  - **FastAPI Backend**: Robust API for document upload and retrieval.
  - **Streamlit Frontend**: Interactive web-based chat interface.
- **Dockerized**: Fully containerized with `docker-compose` for seamless, environment-agnostic deployment.

## Pipeline Flow

```mermaid
flowchart TD
    User[User (Browser)] -->|Uploads PDF| UI[Streamlit UI]
    UI -->|POST /upload| API[FastAPI Backend]
    API -->|Save| Storage[(Local Storage)]
    
    User -->|Asks Question| UI
    UI -->|POST /chat| API
    API -->|Read PDF| Extraction[PDF Extraction]
    Extraction -->|Pages Text| Embedding[Sentence Transformers]
    Embedding -->|Cosine Similarity| Matching[Semantic Match]
    Matching -->|Top Section| Prompt[Prompt Builder]
    Prompt -->|Context + Query| Gemini[Gemini API]
    Gemini -->|Answer| API
    API -->|Response| UI
```

## Setup Instructions

### Prerequisites
- Docker and Docker Compose installed on your machine.
- A Gemini API Key.

### Quick Start (Docker)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AryanSinghGangwar/DocChat_AI.git
   cd DocChat_AI
   ```

2. **Environment Variables:**
   - Copy the `.env.example` file to `.env`:
     ```bash
     cp .env.example .env
     ```
   - Add your Gemini API Key in the `.env` file:
     ```env
     GEMINI_API_KEY=your_gemini_api_key_here
     ```

3. **Run with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
   - **Streamlit UI**: Navigate to [http://localhost:8501](http://localhost:8501)
   - **FastAPI Swagger Docs**: Navigate to [http://localhost:8000/docs](http://localhost:8000/docs)

### Manual Setup (Without Docker)

If you prefer to run the services locally without Docker:

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

## Architecture Details

- **Backend (FastAPI)**: Exposes REST endpoints, handles file storage temporarily, and houses the RAG logic.
- **Frontend (Streamlit)**: Maintains session state (chat history, current document ID) and provides a clean chat UI.
- **Retrieval Engine**: Uses Hugging Face's `sentence-transformers` for dense vector representation and PyTorch for fast cosine similarity computation.
- **Generation Layer**: Leverages Google's Generative AI (Gemini) for reasoning over the retrieved context.
