import os
import shutil
import logging
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import DocChatPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="DocChat AI API", description="API for DocChat AI RAG Pipeline")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize pipeline lazily to avoid loading heavy models if not needed instantly
pipeline: Optional[DocChatPipeline] = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        logger.info("Initializing DocChatPipeline...")
        pipeline = DocChatPipeline()
    return pipeline


class ChatRequest(BaseModel):
    document_id: str
    query: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File uploaded successfully: {file.filename}")
        return {"document_id": file.filename, "message": "File uploaded successfully"}
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="Could not save file")

@app.post("/chat")
async def chat_with_document(request: ChatRequest):
    file_path = os.path.join(UPLOAD_DIR, request.document_id)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Document not found. Please upload it first.")
        
    try:
        pipe = get_pipeline()
        answer = pipe.run(file_path, request.query)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error processing chat: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "healthy"}
