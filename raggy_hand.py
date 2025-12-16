"""
Raggy Hand - Enhanced Recursive RAG System with JSON Index Approach
Handles large documents (300+ pages) without vector databases
Uses LLM-powered semantic routing for efficient retrieval
"""

import pymupdf
import json
import re
import os
import hashlib
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import requests


class LLMClient:
    """Client for interacting with Groq API"""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key is required")
        self.api_key = api_key.strip()
        self.base_url = "https://api.groq.com/openai/v1"

    def generate(self, prompt: str, model: str = "llama-3.3-70b-versatile",
                 max_tokens: int = 2048, temperature: float = 0.7) -> str:
        """Generate a response from the LLM"""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        data = {
            "messages": [{"role": "user", "content": prompt}],
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                raise Exception(f"API Error: {error_msg}")
        except requests.Timeout:
            raise Exception("Request timed out. Please try again.")
        except Exception as e:
            raise Exception(f"LLM generation failed: {str(e)}")


class DocumentProcessor:
    """Extract and process text from PDF documents with advanced chunking"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, int]:
        """Extract text from PDF using PyMuPDF and return text + page count"""
        try:
            doc = pymupdf.open(pdf_path)
            full_text = ""
            page_count = doc.page_count

            print(f"Extracting text from {page_count} pages...")

            for page_num in range(page_count):
                page = doc.load_page(page_num)
                text = page.get_text("text")

                # Add page marker for reference
                full_text += f"\n\n[PAGE {page_num + 1}]\n{text}"

                # Progress indicator
                if (page_num + 1) % 10 == 0:
                    print(f"Processed {page_num + 1}/{page_count} pages...")

            doc.close()
            print(f"✓ Extraction complete: {len(full_text)} characters")
            return full_text.strip(), page_count

        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def sliding_window_chunk(self, text: str, chunk_size: int = 2000,
                            overlap: int = 400) -> List[Dict[str, any]]:
        """
        Split text into overlapping chunks using sliding window approach.
        ENHANCED: Larger overlap (400 words) for better context preservation.
        """
        words = text.split()
        chunks = []
        chunk_id = 0

        i = 0
        while i < len(words):
            # Get chunk_size words
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            # Extract page numbers from this chunk
            page_numbers = self._extract_page_numbers(chunk_text)

            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start_word": i,
                "end_word": i + len(chunk_words),
                "word_count": len(chunk_words),
                "pages": page_numbers,
                "char_count": len(chunk_text)
            })

            chunk_id += 1

            # Move forward by (chunk_size - overlap) to create overlap
            i += (chunk_size - overlap)

        print(f"✓ Created {len(chunks)} chunks with {overlap} word overlap")
        return chunks

    def _extract_page_numbers(self, text: str) -> List[int]:
        """Extract page numbers from text markers"""
        page_pattern = r'\[PAGE (\d+)\]'
        matches = re.findall(page_pattern, text)
        return sorted(list(set(int(p) for p in matches)))

    def convert_to_markdown(self, text: str, is_chinese: bool = False) -> str:
        """Convert extracted text to markdown format using LLM"""

        # For large documents, only convert first few chunks
        preview_text = text[:8000]

        if is_chinese:
            prompt = f"""Convert the following Mandarin text to well-formatted Markdown. 
Preserve the structure, headings, lists, and formatting.
Translate to English if needed, then format as Markdown.

Text:
{preview_text}

Return only the Markdown formatted text."""
        else:
            prompt = f"""Convert the following text to well-formatted Markdown.
Identify headings, subheadings, lists, and code blocks.
Format appropriately using Markdown syntax.

Text:
{preview_text}

Return only the Markdown formatted text."""

        try:
            markdown_text = self.llm_client.generate(prompt, max_tokens=4096, temperature=0.3)
            return markdown_text
        except Exception as e:
            print(f"Markdown conversion skipped: {e}")
            return text


class SmartIndexer:
    """
    Creates a lightweight semantic index for chunks using LLM-generated summaries.
    This replaces vector embeddings with a JSON-based approach.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def create_index(self, chunks: List[Dict], batch_size: int = 2) -> List[Dict]:
        """
        Create an ENHANCED index with richer summaries and metadata.
        Reduced batch size for more detailed processing per chunk.
        """
        index = []
        total_chunks = len(chunks)

        print(f"Creating enhanced semantic index for {total_chunks} chunks...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            batch_indices = self._process_batch(batch)
            index.extend(batch_indices)

            print(f"✓ Indexed {min(i + batch_size, total_chunks)}/{total_chunks} chunks")

        return index

    def _process_batch(self, chunks: List[Dict]) -> List[Dict]:
        """Process chunks with ENHANCED indexing for better retrieval"""
        batch_indices = []

        for chunk in chunks:
            chunk_id = chunk["id"]
            chunk_text = chunk["text"][:3500]  # Increased for more context
            pages = chunk.get("pages", [])

            # ENHANCED prompt for richer metadata
            prompt = f"""Analyze this document chunk deeply and provide comprehensive metadata:

1. DETAILED Summary: Write a 3-4 sentence summary capturing ALL key points
2. Keywords: Extract 10-15 significant keywords and phrases
3. Entities: Identify people, places, organizations, concepts, dates
4. Topics: Main themes and subjects discussed
5. Question Seeds: What questions could this chunk answer? (3-5 questions)

Chunk text:
{chunk_text}

Return ONLY valid JSON:
{{
  "summary": "Detailed 3-4 sentence summary here",
  "keywords": ["keyword1", "keyword2", "...", "keyword15"],
  "entities": ["entity1", "entity2", "entity3"],
  "topics": ["topic1", "topic2", "topic3"],
  "question_seeds": ["What is...?", "How does...?", "When did...?"]
}}"""

            try:
                response = self.llm_client.generate(
                    prompt,
                    max_tokens=800,  # Increased for detailed response
                    temperature=0.2
                )

                # Extract JSON from response
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    metadata = json.loads(json_match.group())
                else:
                    metadata = self._create_fallback_metadata(chunk_text)

                batch_indices.append({
                    "id": chunk_id,
                    "summary": metadata.get("summary", "")[:500],  # Increased length
                    "keywords": metadata.get("keywords", [])[:15],
                    "entities": metadata.get("entities", [])[:15],
                    "topics": metadata.get("topics", [])[:10],
                    "question_seeds": metadata.get("question_seeds", [])[:5],
                    "pages": pages,
                    "preview": chunk_text[:300],  # Longer preview
                    "word_count": chunk.get("word_count", 0)
                })

            except Exception as e:
                print(f"Warning: Failed to index chunk {chunk_id}: {e}")
                batch_indices.append(self._create_fallback_index(chunk_id, chunk_text, pages, chunk))

        return batch_indices

    def _create_fallback_metadata(self, text: str) -> Dict:
        """Create fallback metadata if LLM fails"""
        words = text.split()
        return {
            "summary": text[:400],
            "keywords": list(set(w.lower() for w in words if len(w) > 5))[:15],
            "entities": [],
            "topics": [],
            "question_seeds": []
        }

    def _create_fallback_index(self, chunk_id: int, text: str, pages: List, chunk: Dict) -> Dict:
        """Create fallback index entry"""
        return {
            "id": chunk_id,
            "summary": text[:400],
            "keywords": [],
            "entities": [],
            "topics": [],
            "question_seeds": [],
            "pages": pages,
            "preview": text[:300],
            "word_count": chunk.get("word_count", 0)
        }

    def save_index(self, index: List[Dict], pdf_path: str, cache_dir: str = ".rag_cache"):
        """Save index to JSON file for reuse"""
        os.makedirs(cache_dir, exist_ok=True)

        # Create unique filename based on PDF hash
        pdf_hash = self._hash_file(pdf_path)
        index_path = os.path.join(cache_dir, f"{pdf_hash}_index.json")

        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

        print(f"✓ Index saved to {index_path}")
        return index_path

    def load_index(self, pdf_path: str, cache_dir: str = ".rag_cache") -> Optional[List[Dict]]:
        """Load index from cache if it exists"""
        pdf_hash = self._hash_file(pdf_path)
        index_path = os.path.join(cache_dir, f"{pdf_hash}_index.json")

        if os.path.exists(index_path):
            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                print(f"✓ Loaded cached index from {index_path}")
                return index
            except Exception as e:
                print(f"Warning: Failed to load cached index: {e}")
                return None

        return None

    def _hash_file(self, filepath: str) -> str:
        """Generate hash of file for caching"""
        hasher = hashlib.md5()

        with open(filepath, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        return hasher.hexdigest()


class SemanticRouter:
    """
    Routes queries to relevant chunks using LLM-powered semantic matching.
    This replaces vector similarity with intelligent chunk selection.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def route_query(self, query: str, index: List[Dict], top_k: int = 8) -> List[int]:
        """
        ENHANCED routing with deeper semantic analysis.
        Returns more chunks (top_k=8) for comprehensive answers.
        """
        # Create a richer representation of the index
        index_summary = self._format_index_for_prompt(index)

        prompt = f"""You are an advanced semantic router for document retrieval. Your job is to find ALL relevant chunks that could help answer the user's query.

User Query: {query}

Available Document Chunks (with summaries, keywords, topics, and example questions):
{index_summary}

Instructions:
1. Carefully analyze the user's query to understand what information they need
2. Consider both DIRECT matches (keywords) and SEMANTIC matches (related concepts)
3. Include chunks that provide context, background, or related information
4. Look at summaries, keywords, topics, AND question seeds
5. Return the IDs of the top {top_k} most relevant chunks
6. Order by relevance (most relevant first)
7. Be generous - if a chunk might be relevant, include it

Return ONLY valid JSON with chunk IDs ordered by relevance:
{{"relevant_chunks": [1, 5, 12, 8, 3, 15, 7, 20], "reasoning": "Brief explanation of why these chunks"}}"""

        try:
            response = self.llm_client.generate(
                prompt,
                max_tokens=512,
                temperature=0.15  # Slightly higher for better reasoning
            )

            # Extract JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                chunk_ids = result.get("relevant_chunks", [])
                reasoning = result.get("reasoning", "")

                # Validate chunk IDs
                valid_ids = [cid for cid in chunk_ids if 0 <= cid < len(index)]

                if valid_ids:
                    print(f"✓ Routed to chunks: {valid_ids[:top_k]}")
                    if reasoning:
                        print(f"  Reasoning: {reasoning[:150]}...")
                    return valid_ids[:top_k]

            # Fallback: keyword-based matching
            print("⚠ Using keyword fallback routing")
            return self._fallback_route(query, index, top_k)

        except Exception as e:
            print(f"Warning: Routing failed, using fallback: {e}")
            return self._fallback_route(query, index, top_k)

    def _fallback_route(self, query: str, index: List[Dict], top_k: int) -> List[int]:
        """Fallback routing using simple keyword matching"""
        query_words = set(query.lower().split())
        scores = []

        for item in index:
            score = 0
            text_to_search = f"{item['summary']} {' '.join(item['keywords'])} {' '.join(item.get('topics', []))}"
            text_lower = text_to_search.lower()

            for word in query_words:
                if len(word) > 3 and word in text_lower:
                    score += text_lower.count(word)

            scores.append((item['id'], score))

        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [chunk_id for chunk_id, _ in scores[:top_k]]

    def _format_index_for_prompt(self, index: List[Dict]) -> str:
        """Format index with RICHER information for better routing"""
        lines = []

        for item in index[:100]:  # Limit to first 100 for prompt size
            chunk_id = item["id"]
            summary = item["summary"][:200]
            keywords = ", ".join(item["keywords"][:8])
            topics = ", ".join(item.get("topics", [])[:5])
            questions = " | ".join(item.get("question_seeds", [])[:2])
            pages = f"pg{min(item['pages'])}-{max(item['pages'])}" if item["pages"] else "N/A"

            lines.append(
                f"[{chunk_id}] {pages} | {summary}\n"
                f"    Keywords: {keywords} | Topics: {topics}\n"
                f"    Answers: {questions}"
            )

        return "\n\n".join(lines)


class RAGSystem:
    """
    Enhanced RAG system that handles 300+ page documents without vector databases.
    Uses JSON-based semantic indexing and LLM-powered routing.
    """

    def __init__(self, api_key: str):
        self.llm_client = LLMClient(api_key)
        self.doc_processor = DocumentProcessor(self.llm_client)
        self.indexer = SmartIndexer(self.llm_client)
        self.router = SemanticRouter(self.llm_client)

        # Document storage
        self.full_text = ""
        self.markdown_text = ""
        self.chunks = []
        self.index = []
        self.document_loaded = False
        self.pdf_path = ""
        self.page_count = 0

    def process_document(self, pdf_path: str, convert_chinese: bool = False,
                        force_reindex: bool = False) -> bool:
        """
        Process a PDF document through the enhanced pipeline.
        Uses caching to avoid reprocessing the same document.
        """
        try:
            self.pdf_path = pdf_path

            # Try to load cached index
            if not force_reindex:
                cached_index = self.indexer.load_index(pdf_path)
                if cached_index:
                    self.index = cached_index
                    print("✓ Using cached index")

                    # Still need to load chunks for retrieval
                    print("Loading document text...")
                    self.full_text, self.page_count = self.doc_processor.extract_text_from_pdf(pdf_path)
                    self.chunks = self.doc_processor.sliding_window_chunk(
                        self.full_text,
                        chunk_size=3000,
                        overlap=300
                    )

                    self.document_loaded = True
                    return True

            # Full processing pipeline
            print("\n=== Starting Document Processing ===\n")

            # Step 1: Extract text
            print("Step 1/4: Extracting text from PDF...")
            self.full_text, self.page_count = self.doc_processor.extract_text_from_pdf(pdf_path)

            if not self.full_text.strip():
                raise Exception("No text extracted from PDF")

            # Step 2: Create chunks with ENHANCED sliding window
            print("\nStep 2/4: Creating overlapping chunks with enhanced context preservation...")
            self.chunks = self.doc_processor.sliding_window_chunk(
                self.full_text,
                chunk_size=2000,  # Smaller chunks for better granularity
                overlap=400       # Larger overlap (20%) for context continuity
            )

            # Step 3: Create ENHANCED semantic index
            print("\nStep 3/4: Creating deep semantic index with LLM...")
            print("  (This generates rich metadata: summaries, keywords, topics, questions)")
            self.index = self.indexer.create_index(self.chunks, batch_size=2)

            # Step 4: Save index to cache
            print("\nStep 4/4: Saving index to cache...")
            self.indexer.save_index(self.index, pdf_path)

            # Optional: Convert to markdown (only preview)
            if convert_chinese:
                print("\nConverting to Markdown...")
                self.markdown_text = self.doc_processor.convert_to_markdown(
                    self.full_text[:10000],
                    is_chinese=True
                )

            self.document_loaded = True

            print("\n=== Document Processing Complete ===")
            print(f"✓ Pages: {self.page_count}")
            print(f"✓ Chunks: {len(self.chunks)}")
            print(f"✓ Index entries: {len(self.index)}")

            return True

        except Exception as e:
            print(f"\n✗ Document processing error: {e}")
            self.document_loaded = False
            return False

    def query(self, user_query: str, model: str = "llama-3.3-70b-versatile",
              top_k: int = 8) -> str:
        """
        ENHANCED query processing with deeper context and comprehensive retrieval.
        Uses top_k=8 chunks for more complete answers.
        """
        if not self.document_loaded:
            return "Error: No document loaded. Please upload a PDF first."

        try:
            print(f"\n=== Processing Query ===")
            print(f"Query: {user_query}\n")

            # Step 1: Route to relevant chunks with enhanced matching
            print("Step 1/3: Finding relevant chunks with semantic routing...")
            relevant_chunk_ids = self.router.route_query(user_query, self.index, top_k=top_k)

            if not relevant_chunk_ids:
                return "I couldn't find relevant information in the document for your query. Please try rephrasing or asking a more specific question."

            # Step 2: Assemble RICH context from chunks
            print(f"Step 2/3: Assembling context from {len(relevant_chunk_ids)} chunks...")
            context_parts = []
            total_context_length = 0

            for chunk_id in relevant_chunk_ids:
                if chunk_id < len(self.chunks):
                    chunk = self.chunks[chunk_id]
                    index_item = self.index[chunk_id]

                    pages_str = f"Pages {min(index_item['pages'])}-{max(index_item['pages'])}" if index_item['pages'] else "N/A"

                    # Include MORE context per chunk
                    chunk_context = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHUNK {chunk_id} | {pages_str}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 Summary: {index_item['summary']}

 Keywords: {', '.join(index_item['keywords'][:10])}

 Topics: {', '.join(index_item.get('topics', [])[:5])}

 Full Content:
{chunk['text'][:5000]}
"""

                    context_parts.append(chunk_context)
                    total_context_length += len(chunk_context)

                    # Stop if context gets too large (to fit in LLM window)
                    if total_context_length > 25000:
                        print(f"  Context limit reached at {len(context_parts)} chunks")
                        break

            context = "\n\n".join(context_parts)

            print(f"  Assembled {len(context_parts)} chunks ({len(context)} characters)")

            # Step 3: Generate comprehensive answer
            print("Step 3/3: Generating comprehensive answer...")

            final_prompt = f"""You are an expert document analyst. Answer the user's question based STRICTLY on the provided document excerpts.

CRITICAL INSTRUCTIONS:
1. Use ONLY information from the provided context below
2. If information is in the context, provide a DETAILED and COMPREHENSIVE answer
3. Cite specific page numbers when referencing information
4. If the answer is NOT in the context, explicitly state: "This information is not found in the provided document sections"
5. Synthesize information across multiple chunks if needed
6. Be thorough - don't just give brief answers, provide full explanations
7. Quote relevant passages when appropriate (use quotation marks)

DOCUMENT CONTEXT:
{context}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

USER QUESTION: {user_query}

COMPREHENSIVE ANSWER (with page citations):"""

            answer = self.llm_client.generate(
                final_prompt,
                model=model,
                max_tokens=3000,  # Increased for detailed answers
                temperature=0.3
            )

            print("✓ Answer generated\n")

            # Add metadata footer
            pages_covered = set()
            for chunk_id in relevant_chunk_ids[:len(context_parts)]:
                if chunk_id < len(self.index):
                    pages_covered.update(self.index[chunk_id]['pages'])

            if pages_covered:
                page_range = f"Pages {min(pages_covered)}-{max(pages_covered)}"
                answer += f"\n\n──────────────────────────────────\n📖 Information retrieved from: {page_range}\n📦 Analyzed {len(context_parts)} relevant sections"

            return answer

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg

    def is_document_loaded(self) -> bool:
        """Check if a document is currently loaded"""
        return self.document_loaded

    def get_full_text(self) -> str:
        """Get the full extracted text"""
        return self.markdown_text if self.markdown_text else self.full_text

    def get_stats(self) -> Dict:
        """Get comprehensive statistics about the loaded document"""
        if not self.document_loaded:
            return {
                "pages": 0,
                "chunks": 0,
                "index_entries": 0,
                "total_words": 0,
                "document_loaded": False,
                "pdf_path": ""
            }

        return {
            "pages": self.page_count,
            "chunks": len(self.chunks),
            "index_entries": len(self.index),
            "total_words": len(self.full_text.split()) if self.full_text else 0,
            "total_chars": len(self.full_text) if self.full_text else 0,
            "document_loaded": self.document_loaded,
            "pdf_path": self.pdf_path,
            "avg_chunk_size": sum(c.get('word_count', 0) for c in self.chunks) // len(self.chunks) if self.chunks else 0
        }

    def clear_cache(self, cache_dir: str = ".rag_cache"):
        """Clear all cached indices"""
        try:
            if os.path.exists(cache_dir):
                for file in os.listdir(cache_dir):
                    os.remove(os.path.join(cache_dir, file))
                print(f"✓ Cache cleared: {cache_dir}")
        except Exception as e:
            print(f"Warning: Failed to clear cache: {e}")