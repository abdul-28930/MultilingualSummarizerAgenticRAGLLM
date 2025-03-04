import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client
import numpy as np
from datetime import datetime
import logging
import json
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwoPassRAG:
    def __init__(self):
        """Initialize the RAG system with Supabase and OpenAI"""
        load_dotenv()
        
        # Initialize OpenAI
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Initialize Supabase
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        if not (self.supabase_url and self.supabase_key):
            raise ValueError("Supabase credentials not found in environment variables")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        
        # Initialize tokenizer for text chunking
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings from OpenAI API"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            raise

    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Split text into chunks based on token count"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), chunk_size):
            chunk_tokens = tokens[i:i + chunk_size]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk_text)
            
        return chunks

    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using GPT-4"""
        try:
            prompt = f"""Extract the most important keywords from the following text. 
            Return only a comma-separated list of keywords, no other text.
            
            Text:
            {text}"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a keyword extraction assistant. Return only comma-separated keywords, no other text."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            keywords = response.choices[0].message.content.strip().split(',')
            return [k.strip() for k in keywords]
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []

    def store_document(self, text: str, source: str = None, chunk_size: int = 500):
        """Store document chunks in Supabase with embeddings"""
        try:
            chunks = self.chunk_text(text, chunk_size)
            total_chunks = len(chunks)
            
            for i, chunk in enumerate(chunks):
                # Get embedding for the chunk
                embedding = self.get_embedding(chunk)
                
                # Extract keywords for the chunk
                keywords = self.extract_keywords(chunk)
                
                # Store in Supabase
                self.supabase.table('documents').insert({
                    'content': chunk,
                    'embedding': embedding,
                    'metadata': {'source': source} if source else {},
                    'keywords': keywords,
                    'chunk_index': i,
                    'total_chunks': total_chunks,
                    'created_at': datetime.now().isoformat()
                }).execute()
            
            logger.info(f"Stored {len(chunks)} chunks in Supabase")
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            raise

    def retrieve_relevant_chunks(self, query: str, match_count: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from Supabase based on query embedding"""
        try:
            query_embedding = self.get_embedding(query)
            
            # Get chunks by similarity search
            chunks = self.supabase.table('documents').select(
                'id',
                'content',
                'metadata',
                f'1 - (embedding <=> cast({query_embedding} as vector)) as similarity'
            ).order(
                'similarity',
                desc=True
            ).limit(match_count).execute()
            
            return chunks.data if chunks.data else []
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            raise

    def generate_first_pass(self, query: str, context: List[Dict[str, Any]]) -> str:
        """First pass: Generate initial response using retrieved context"""
        try:
            prompt = f"""Given the following context and question, generate a comprehensive initial response.
            Be factual and only use information from the provided context.
            
            Context:
            {' '.join([chunk['content'] for chunk in context])}
            
            Question:
            {query}
            
            Initial Response:"""
            
            response = self.client.chat_completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates accurate responses based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in first pass generation: {e}")
            return ""

    def generate_second_pass(self, query: str, first_pass_response: str, context: List[Dict[str, Any]]) -> str:
        """Second pass: Refine and improve the initial response"""
        try:
            prompt = f"""Review and improve the following initial response. Ensure accuracy, clarity, and completeness.
            If the initial response contains any inaccuracies based on the context, correct them.
            Add any important missing information from the context.
            
            Original Question:
            {query}
            
            Context:
            {' '.join([chunk['content'] for chunk in context])}
            
            Initial Response:
            {first_pass_response}
            
            Improved Response:"""
            
            response = self.client.chat_completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that reviews and improves responses for accuracy and completeness."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in second pass generation: {e}")
            return first_pass_response

    def query(self, query: str) -> Dict[str, str]:
        """Main query function implementing the two-pass architecture"""
        try:
            # Retrieve relevant chunks
            relevant_chunks = self.retrieve_relevant_chunks(query)
            
            if not relevant_chunks:
                return {
                    "error": "No relevant context found for the query",
                    "first_pass": "",
                    "final_response": ""
                }
            
            # First pass: Generate initial response
            first_pass_response = self.generate_first_pass(query, relevant_chunks)
            
            # Second pass: Refine and improve
            final_response = self.generate_second_pass(query, first_pass_response, relevant_chunks)
            
            return {
                "first_pass": first_pass_response,
                "final_response": final_response
            }
            
        except Exception as e:
            logger.error(f"Error in query processing: {e}")
            return {
                "error": str(e),
                "first_pass": "",
                "final_response": ""
            }

def main():
    # Initialize RAG system
    rag = TwoPassRAG()
    
    # Example usage
    print("Two-Pass RAG System")
    print("------------------")
    
    while True:
        print("\nOptions:")
        print("1. Store new document")
        print("2. Query existing documents")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == "1":
            print("\nEnter the document text (press Enter twice to finish):")
            lines = []
            while True:
                line = input()
                if line.strip() == "":
                    break
                lines.append(line)
            
            text = "\n".join(lines)
            
            if text.strip():
                source = input("Enter document source (optional): ") or "unknown"
                
                try:
                    rag.store_document(text, source)
                    print("Document stored successfully!")
                except Exception as e:
                    print(f"Error storing document: {e}")
            
        elif choice == "2":
            query = input("\nEnter your query: ")
            if query.strip():
                try:
                    result = rag.query(query)
                    
                    if "error" in result:
                        print(f"\nError: {result['error']}")
                    else:
                        print("\nFirst Pass Response:")
                        print(result["first_pass"])
                        print("\nFinal Refined Response:")
                        print(result["final_response"])
                except Exception as e:
                    print(f"Error processing query: {e}")
            
        elif choice == "3":
            print("\nGoodbye!")
            break
        
        else:
            print("\nInvalid choice. Please try again.")

if __name__ == "__main__":
    main()
