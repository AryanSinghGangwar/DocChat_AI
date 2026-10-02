import os
import json
import logging
import argparse
from typing import List, Dict, Tuple, Any, Optional

import pdfplumber
import requests
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocChatPipeline:
    """
    A pipeline to extract text from a PDF, find relevant sections using embeddings,
    and answer questions using the Gemini API.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the pipeline and load the embedding model."""
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key or self.api_key == "******" or self.api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY environment variable not set. Please check your .env file.")
            
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={self.api_key}"
        
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.embedder = SentenceTransformer(model_name)

    def extract_pdf_text(self, file_path: str) -> List[Tuple[int, str]]:
        """Extract text from each page of the PDF."""
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
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def find_relevant_section(self, pages_text: List[Tuple[int, str]], query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """Find the most relevant sections for the given query."""
        if not pages_text:
            logger.warning("No text extracted from the document to search.")
            return []

        logger.info(f"Finding top {top_k} relevant sections for query: '{query}'")
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
        """Call the Gemini API with the given prompt."""
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }

        try:
            logger.info("Sending request to Gemini API...")
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            
            data = response.json()
            return data['candidates'][0]['content']['parts'][0]['text']
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if 'response' in locals() and response is not None and hasattr(response, 'text'):
                logger.error(f"Response details: {response.text}")
            return f"Error connecting to Gemini API: {e}"
        except (KeyError, IndexError) as e:
            logger.error(f"Unexpected API response format: {e}")
            return "Error parsing Gemini API response."

    def explain_text(self, text: str, question: str) -> str:
        """Generate an explanation based on the text and user question."""
        prompt = f"""
A user asked: "{question}"

Here is an excerpt from a document that might help:

{text}

Please explain the relevant information clearly and thoroughly in approximately 250 Words.
"""
        return self.call_gemini_api(prompt)

    def run(self, pdf_path: str, user_question: str) -> Optional[str]:
        """Run the full pipeline."""
        try:
            pages = self.extract_pdf_text(pdf_path)
            if not pages:
                return "No text could be extracted from the PDF."

            relevant_sections = self.find_relevant_section(pages, user_question, top_k=1)
            if not relevant_sections:
                return "Could not find any relevant sections in the document."

            relevant = relevant_sections[0]
            logger.info(f"Relevant content found on page {relevant['page']} (score: {relevant['score']:.4f})")
            
            # Log truncated relevant text to debug/info
            truncated_text = relevant['text'][:500] + "..." if len(relevant['text']) > 500 else relevant['text']
            logger.debug(f"Relevant text snippet:\n{truncated_text}")

            explanation = self.explain_text(relevant['text'], user_question)
            return explanation

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="DocChat AI - Ask questions about your PDF documents.")
    parser.add_argument("--pdf", type=str, help="Path to the PDF file")
    parser.add_argument("--query", type=str, help="Question to ask about the PDF")
    
    args = parser.parse_args()
    
    pdf_path = args.pdf
    query = args.query
    
    # Fallback to interactive mode if arguments are not provided
    if not pdf_path:
        pdf_path = input("Enter the path to the PDF file: ").strip()
    
    if not os.path.exists(pdf_path) or not pdf_path.lower().endswith('.pdf'):
        logger.error(f"Invalid PDF file path: {pdf_path}")
        print("Error: Please provide a valid, existing PDF file.")
        return

    if not query:
        query = input("Enter your question: ").strip()
        
    if not query:
        logger.error("Question cannot be empty.")
        return

    try:
        pipeline = DocChatPipeline()
        print("\n" + "="*50)
        print(f"Processing document: {pdf_path}")
        print(f"Question: {query}")
        print("="*50 + "\n")
        
        answer = pipeline.run(pdf_path, query)
        
        if answer:
            print("\n" + "="*20 + " ANSWER " + "="*20)
            print(answer)
            print("="*48 + "\n")
        else:
            print("\nFailed to generate an answer. Please check the logs.\n")
            
    except Exception as e:
        print(f"\nInitialization Error: {e}")
        print("Please ensure your .env file is set up correctly with GEMINI_API_KEY.")

if __name__ == "__main__":
    main()
