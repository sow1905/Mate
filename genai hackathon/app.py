import streamlit as st
import PyMuPDF  # fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
import json
import os
from typing import List, Dict, Tuple
import tempfile
import re
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="StudyMate - AI Academic Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        text-align: center;
    }
    .main-header p {
        color: white;
        text-align: center;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    .document-card {
        background: #f8f9ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .answer-card {
        background: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .source-card {
        background: #fff8f0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #ff9800;
        margin: 0.5rem 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

class StudyMate:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.api_key = None
        self.project_id = None
        
    def initialize_embedding_model(self):
        """Initialize the sentence transformer model for embeddings"""
        if self.embedding_model is None:
            with st.spinner("Loading embedding model..."):
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.embedding_model
    
    def extract_text_from_pdf(self, pdf_file) -> List[Dict]:
        """Extract text from PDF using PyMuPDF"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            
            # Extract text using PyMuPDF
            doc = fitz.open(tmp_file_path)
            extracted_data = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():  # Only add non-empty pages
                    extracted_data.append({
                        'page': page_num + 1,
                        'text': text,
                        'source': pdf_file.name
                    })
            
            doc.close()
            os.unlink(tmp_file_path)  # Clean up temp file
            
            return extracted_data
            
        except Exception as e:
            st.error(f"Error extracting text from {pdf_file.name}: {str(e)}")
            return []
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def process_documents(self, pdf_files) -> bool:
        """Process uploaded PDF documents"""
        try:
            self.chunks = []
            self.chunk_metadata = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(pdf_files)
            
            for file_idx, pdf_file in enumerate(pdf_files):
                status_text.text(f"Processing {pdf_file.name}...")
                
                # Extract text from PDF
                extracted_data = self.extract_text_from_pdf(pdf_file)
                
                # Process each page
                for page_data in extracted_data:
                    # Chunk the text
                    page_chunks = self.chunk_text(page_data['text'])
                    
                    for chunk_idx, chunk in enumerate(page_chunks):
                        self.chunks.append(chunk)
                        self.chunk_metadata.append({
                            'source': page_data['source'],
                            'page': page_data['page'],
                            'chunk_id': chunk_idx
                        })
                
                # Update progress
                progress_bar.progress((file_idx + 1) / total_files)
            
            status_text.text("Creating embeddings...")
            
            # Initialize embedding model
            self.initialize_embedding_model()
            
            # Create embeddings for all chunks
            embeddings = self.embedding_model.encode(self.chunks)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings.astype('float32'))
            
            progress_bar.progress(1.0)
            status_text.text("✅ Documents processed successfully!")
            
            return True
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            return False
    
    def search_relevant_chunks(self, query: str, top_k: int = 3) -> List[Tuple[str, Dict, float]]:
        """Search for relevant chunks using FAISS"""
        if self.index is None or not self.chunks:
            return []
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                results.append((
                    self.chunks[idx],
                    self.chunk_metadata[idx],
                    float(score)
                ))
        
        return results
    
    def generate_answer_watson(self, query: str, context: str) -> str:
        """Generate answer using IBM Watsonx (Mixtral model)"""
        if not self.api_key or not self.project_id:
            return "❌ IBM Watsonx credentials not configured. Please set up your API key and project ID in the sidebar."
        
        try:
            # IBM Watsonx API endpoint (adjust based on your region)
            url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
            
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            prompt = f"""Based on the following context from academic documents, please provide a comprehensive and accurate answer to the question. If the context doesn't contain enough information to fully answer the question, please state that clearly.

Context:
{context}

Question: {query}

Answer:"""
            
            data = {
                "input": prompt,
                "parameters": {
                    "decoding_method": "greedy",
                    "max_new_tokens": 500,
                    "temperature": 0.1,
                    "repetition_penalty": 1.1
                },
                "model_id": "mistralai/mixtral-8x7b-instruct-v01",
                "project_id": self.project_id
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                return result['results'][0]['generated_text']
            else:
                return f"❌ Error from IBM Watsonx: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"❌ Error generating answer: {str(e)}"
    
    def generate_answer_local(self, query: str, context: str) -> str:
        """Generate answer using local processing (fallback)"""
        # Simple template-based response when Watson isn't available
        sentences = context.split('.')
        relevant_sentences = [s.strip() for s in sentences if any(word.lower() in s.lower() for word in query.split()) and len(s.strip()) > 20]
        
        if relevant_sentences:
            answer = "Based on the provided documents:\n\n"
            answer += ". ".join(relevant_sentences[:3]) + "."
            return answer
        else:
            return "I found relevant content in your documents, but I couldn't identify specific information that directly answers your question. Please try rephrasing your question or asking about different aspects of the topic."

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📚 StudyMate</h1>
        <p>AI-Powered Academic Assistant - Upload PDFs and ask questions!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize StudyMate
    if 'studymate' not in st.session_state:
        st.session_state.studymate = StudyMate()
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # IBM Watsonx Configuration
        st.subheader("IBM Watsonx Setup")
        api_key = st.text_input("API Key", type="password", help="Your IBM Watsonx API key")
        project_id = st.text_input("Project ID", help="Your IBM Watsonx project ID")
        
        if api_key and project_id:
            st.session_state.studymate.api_key = api_key
            st.session_state.studymate.project_id = project_id
            st.success("✅ Watsonx configured")
        else:
            st.warning("⚠️ Enter Watsonx credentials for full functionality")
        
        st.divider()
        
        # Document Upload
        st.subheader("📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload your study materials (textbooks, notes, papers)"
        )
        
        if uploaded_files:
            if st.button("🔄 Process Documents", type="primary"):
                if st.session_state.studymate.process_documents(uploaded_files):
                    st.session_state.processed_files = [f.name for f in uploaded_files]
                    st.rerun()
        
        # Display processed files
        if st.session_state.processed_files:
            st.subheader("📚 Processed Documents")
            for file_name in st.session_state.processed_files:
                st.markdown(f'<div class="document-card">📄 {file_name}</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Settings
        st.subheader("⚙️ Settings")
        search_results = st.slider("Search Results", 1, 10, 3, help="Number of relevant chunks to retrieve")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Ask Questions")
        
        if not st.session_state.processed_files:
            st.info("👆 Please upload and process PDF documents in the sidebar to get started!")
        else:
            # Question input
            question = st.text_input(
                "What would you like to know?",
                placeholder="e.g., What are the main principles of machine learning?",
                key="question_input"
            )
            
            col_ask, col_clear = st.columns([3, 1])
            with col_ask:
                ask_button = st.button("🔍 Ask Question", type="primary", use_container_width=True)
            with col_clear:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.question_input = ""
                    st.rerun()
            
            if ask_button and question:
                with st.spinner("🔍 Searching documents and generating answer..."):
                    # Search for relevant chunks
                    relevant_chunks = st.session_state.studymate.search_relevant_chunks(
                        question, top_k=search_results
                    )
                    
                    if relevant_chunks:
                        # Prepare context
                        context = "\n\n".join([chunk for chunk, _, _ in relevant_chunks])
                        
                        # Generate answer
                        if st.session_state.studymate.api_key:
                            answer = st.session_state.studymate.generate_answer_watson(question, context)
                        else:
                            answer = st.session_state.studymate.generate_answer_local(question, context)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': question,
                            'answer': answer,
                            'sources': relevant_chunks,
                            'timestamp': datetime.now().strftime("%H:%M:%S")
                        })
                        
                        # Clear input
                        st.session_state.question_input = ""
                        st.rerun()
                    else:
                        st.error("❌ No relevant information found in your documents for this question.")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("📝 Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"❓ {chat['question'][:60]}..." if len(chat['question']) > 60 else f"❓ {chat['question']}", expanded=(i==0)):
                    st.markdown(f"**Question:** {chat['question']}")
                    st.markdown(f'<div class="answer-card"><strong>Answer:</strong><br>{chat["answer"]}</div>', unsafe_allow_html=True)
                    
                    if chat['sources']:
                        st.markdown("**📚 Sources:**")
                        for j, (chunk, metadata, score) in enumerate(chat['sources']):
                            st.markdown(f'''
                            <div class="source-card">
                                <strong>Source {j+1}:</strong> {metadata["source"]} (Page {metadata["page"]}) - Relevance: {score:.2f}<br>
                                <em>{chunk[:200]}...</em>
                            </div>
                            ''', unsafe_allow_html=True)
                    
                    st.markdown(f"*Asked at {chat['timestamp']}*")
    
    with col2:
        st.subheader("📊 Statistics")
        
        if st.session_state.processed_files:
            # Document stats
            st.metric("📄 Documents Processed", len(st.session_state.processed_files))
            st.metric("📝 Text Chunks", len(st.session_state.studymate.chunks))
            st.metric("💬 Questions Asked", len(st.session_state.chat_history))
            
            # Recent activity
            if st.session_state.chat_history:
                st.subheader("🕒 Recent Questions")
                for chat in st.session_state.chat_history[-3:]:
                    st.markdown(f"• {chat['question'][:40]}..." if len(chat['question']) > 40 else f"• {chat['question']}")
        
        # Instructions
        st.subheader("ℹ️ How to Use")
        st.markdown("""
        1. **Configure** IBM Watsonx credentials in the sidebar
        2. **Upload** your PDF study materials
        3. **Process** the documents
        4. **Ask** natural language questions
        5. **Get** contextual answers with sources
        """)
        
        st.subheader("💡 Tips")
        st.markdown("""
        - Ask specific questions for better results
        - Use keywords from your documents
        - Try different phrasings if needed
        - Check the sources for verification
        """)

if __name__ == "__main__":
    main()