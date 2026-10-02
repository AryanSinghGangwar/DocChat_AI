import os
import json
import logging
from typing import List, Dict, Tuple, Any, Optional

import pdfplumber
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class DocChatPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "******" or self.api_key == "your_gemini_api_key_here":
            logger.warning("GEMINI_API_KEY environment variable not set or is default.")
            
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedder = SentenceTransformer(model_name)

    def extract_pdf_text(self, file_path: str) -> List[Tuple[int, str]]:
        logger.info(f"Extracting text from PDF: {file_path}")
        pages_text = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_text.append((i + 1, text))
            logger.info(f"Successfully extracted {len(pages_text)} pages with text.")
            return pages_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def find_relevant_section(self, pages_text: List[Tuple[int, str]], query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        if not pages_text:
            return []

        texts = [text for _, text in pages_text]
        doc_embeddings = self.embedder.encode(texts, convert_to_tensor=True)
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)

        similarities = util.pytorch_cos_sim(query_embedding, doc_embeddings)[0]
        top_hits = similarities.topk(min(top_k, len(texts)))

        results = []
        for score, idx in zip(top_hits.values, top_hits.indices):
            page_num, text = pages_text[idx]
            results.append({
                "page": page_num,
                "score": float(score),
                "text": text
            })
        return results

    def call_gemini_api(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            logger.error(f"API request failed: {e}")
            return f"Error connecting to Gemini API: {str(e)}"

    def explain_text(self, text: str, question: str) -> str:
        prompt = f"""
A user asked: "{question}"

Here is an excerpt from a document that might help:

{text}

Please explain the relevant information clearly and thoroughly in approximately 250 Words.
"""
        return self.call_gemini_api(prompt)

    def run(self, pdf_path: str, user_question: str) -> str:
        try:
            pages = self.extract_pdf_text(pdf_path)
            if not pages:
                return "No text could be extracted from the PDF."

            relevant_sections = self.find_relevant_section(pages, user_question, top_k=1)
            if not relevant_sections:
                return "Could not find any relevant sections in the document."

            relevant = relevant_sections[0]
            logger.info(f"Relevant content found on page {relevant['page']} (score: {relevant['score']:.4f})")
            
            explanation = self.explain_text(relevant['text'], user_question)
            return explanation

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return f"An error occurred while processing your request: {str(e)}"
