"""
Integration tests using Cohere embeddings for real-world testing
"""

import pytest
import os
import time
from typing import List, Dict
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from vector_db_api.main import app


# Cohere API key - replace with your actual key
COHERE_API_KEY = "pa6sRhnVAedMVClPAwoCvC1MjHKEwjtcGSTjWRMd"

def get_cohere_client():
    """Get Cohere client if available"""
    try:
        import cohere
        # Use constant key or fallback to environment variable
        api_key = COHERE_API_KEY or os.getenv("COHERE_API_KEY")
        if not api_key:
            return None
        return cohere.Client(api_key)
    except ImportError:
        return None


def generate_cohere_embeddings(texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    """Generate embeddings using Cohere"""
    client = get_cohere_client()
    if not client:
        raise pytest.skip("Cohere client not available")
    
    response = client.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type=input_type
    )
    return response.embeddings


@pytest.mark.cohere
@pytest.mark.integration
class TestCohereEmbeddingsIntegration:
    """Integration tests using Cohere embeddings for realistic testing"""
    
    @pytest.mark.cohere
    def test_cohere_embedding_generation(self):
        """Test that we can generate embeddings using Cohere"""
        test_texts = [
            "The quick brown fox jumps over the lazy dog",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language for data science"
        ]
        
        # Generate embeddings
        embeddings = generate_cohere_embeddings(test_texts, "search_document")
        
        # Verify embeddings
        assert len(embeddings) == len(test_texts)
        assert len(embeddings[0]) == 1024  # Cohere embed-english-v3.0 dimension
        
        # Verify embeddings are different for different texts
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]
        
        return embeddings
    
    @pytest.mark.cohere
    def test_end_to_end_with_cohere_embeddings(self):
        """Test complete workflow with real Cohere embeddings"""
        with TestClient(app) as client:
            # 1. Create a library with Cohere embedding dimensions
            library_data = {
                "name": "Cohere Test Library",
                "description": "Library for testing with Cohere embeddings",
                "embedding_dim": 1024,  # Cohere embed-english-v3.0 dimension
                "index_config": {"type": "flat"}
            }
            create_response = client.post("/libraries", json=library_data)
            assert create_response.status_code == 201
            library = create_response.json()
            library_id = library["id"]
            
            try:
                # 2. Create a document
                document_data = {
                    "title": "AI and Machine Learning",
                    "content": "This document discusses artificial intelligence and machine learning concepts."
                }
                doc_response = client.post(f"/libraries/{library_id}/documents", json=document_data)
                assert doc_response.status_code == 201
                document = doc_response.json()
                document_id = document["id"]
                
                # 3. Prepare text chunks for embedding
                text_chunks = [
                    "Artificial intelligence is the simulation of human intelligence in machines.",
                    "Machine learning is a subset of AI that focuses on algorithms that can learn from data.",
                    "Deep learning uses neural networks with multiple layers to model complex patterns.",
                    "Natural language processing helps computers understand and generate human language.",
                    "Computer vision enables machines to interpret and understand visual information."
                ]
                
                # 4. Generate embeddings using Cohere
                print(f"Generating embeddings for {len(text_chunks)} chunks...")
                embeddings = generate_cohere_embeddings(text_chunks, "search_document")
                
                # 5. Create chunks with real embeddings
                chunk_data = {
                    "chunks": []
                }
                
                for i, (text, embedding) in enumerate(zip(text_chunks, embeddings)):
                    chunk_data["chunks"].append({
                        "text": text,
                        "embedding": embedding,
                        "position": i,
                        "metadata": {
                            "token_count": len(text.split()),
                            "tags": ["ai", "ml", "test"]
                        }
                    })
                
                chunks_response = client.post(
                    f"/libraries/{library_id}/documents/{document_id}/chunks:bulk", 
                    json=chunk_data
                )
                assert chunks_response.status_code == 201
                chunks_result = chunks_response.json()
                assert "chunk_ids" in chunks_result
                assert len(chunks_result["chunk_ids"]) == len(text_chunks)
                
                # 6. Test semantic search with Cohere embeddings
                search_queries = [
                    "What is artificial intelligence?",
                    "How do neural networks work?",
                    "What is computer vision?"
                ]
                
                for query in search_queries:
                    # Generate query embedding
                    query_embedding = generate_cohere_embeddings([query], "search_query")[0]
                    
                    # Search using the query embedding
                    search_data = {
                        "query_embedding": query_embedding,
                        "k": 3
                    }
                    search_response = client.post(f"/libraries/{library_id}/search", json=search_data)
                    assert search_response.status_code == 200
                    search_response_data = search_response.json()
                    search_results = search_response_data["results"]
                    
                    # Verify we get relevant results
                    assert len(search_results) > 0
                    assert len(search_results) <= 3
                    
                    # Check that results have similarity scores
                    for result in search_results:
                        assert "chunk_id" in result
                        assert "score" in result
                        assert "text" in result
                        assert isinstance(result["score"], (int, float))
                        assert result["score"] >= 0  # Similarity scores should be non-negative
                    
                    print(f"Query: '{query}'")
                    print(f"Top result: '{search_results[0]['text'][:100]}...' (score: {search_results[0]['score']:.4f})")
                
                # 7. Test similarity between related concepts
                ai_text = "Artificial intelligence is the simulation of human intelligence in machines."
                ml_text = "Machine learning is a subset of AI that focuses on algorithms that can learn from data."
                
                # Generate embeddings for both
                embeddings = generate_cohere_embeddings([ai_text, ml_text], "search_document")
                ai_embedding = embeddings[0]
                ml_embedding = embeddings[1]
                
                # Calculate cosine similarity manually
                import numpy as np
                ai_np = np.array(ai_embedding)
                ml_np = np.array(ml_embedding)
                cosine_sim = np.dot(ai_np, ml_np) / (np.linalg.norm(ai_np) * np.linalg.norm(ml_np))
                
                print(f"Cosine similarity between AI and ML texts: {cosine_sim:.4f}")
                
                # AI and ML should be similar concepts
                assert cosine_sim > 0.5, f"AI and ML should be similar, got similarity: {cosine_sim}"
                
            finally:
                # 8. Clean up - delete the library
                delete_response = client.delete(f"/libraries/{library_id}")
                assert delete_response.status_code == 204
    
    @pytest.mark.cohere
    def test_embedding_consistency(self):
        """Test that Cohere embeddings are consistent for the same text"""
        test_text = "This is a test for embedding consistency."
        
        # Generate embeddings multiple times
        embeddings = []
        for _ in range(3):
            embedding = generate_cohere_embeddings([test_text], "search_document")[0]
            embeddings.append(embedding)
        
        # Embeddings should be identical (deterministic)
        for i in range(1, len(embeddings)):
            assert embeddings[0] == embeddings[i], "Embeddings should be consistent for the same text"
    
    @pytest.mark.cohere
    def test_embedding_dimensions(self):
        """Test that embeddings have the correct dimensions"""
        test_texts = [
            "Short text",
            "This is a much longer text that contains more words and should still produce the same dimensional embedding as the short text above."
        ]
        
        embeddings = generate_cohere_embeddings(test_texts, "search_document")
        
        # All embeddings should have the same dimension
        for embedding in embeddings:
            assert len(embedding) == 1024  # Cohere embed-english-v3.0 dimension
        
        # Different length texts should produce same dimensional embeddings
        assert len(embeddings[0]) == len(embeddings[1])
    
    @pytest.mark.cohere
    def test_semantic_similarity_ranking(self):
        """Test that semantically similar texts have higher similarity scores"""
        with TestClient(app) as client:
            # Create library
            library_data = {
                "name": "Similarity Test Library",
                "description": "Library for testing semantic similarity",
                "embedding_dim": 1024,  # Cohere embed-english-v3.0 dimension
                "index_config": {"type": "flat"}
            }
            create_response = client.post("/libraries", json=library_data)
            assert create_response.status_code == 201
            library = create_response.json()
            library_id = library["id"]
            
            try:
                # Create document
                document_data = {
                    "title": "Similarity Test Document",
                    "content": "Testing semantic similarity with various topics."
                }
                doc_response = client.post(f"/libraries/{library_id}/documents", json=document_data)
                assert doc_response.status_code == 201
                document = doc_response.json()
                document_id = document["id"]
                
                # Create chunks with diverse topics
                topics = [
                    "Dogs are loyal pets that make great companions.",
                    "Cats are independent animals that enjoy climbing.",
                    "Machine learning algorithms can process large datasets.",
                    "Deep learning uses neural networks for pattern recognition.",
                    "Cooking requires patience and attention to detail.",
                    "Baking is a precise science that needs exact measurements."
                ]
                
                # Generate embeddings
                embeddings = generate_cohere_embeddings(topics, "search_document")
                
                # Create chunks
                chunk_data = {
                    "chunks": [
                        {
                            "text": text,
                            "embedding": embedding,
                            "position": i,
                            "metadata": {"tags": ["test"]}
                        }
                        for i, (text, embedding) in enumerate(zip(topics, embeddings))
                    ]
                }
                
                chunks_response = client.post(
                    f"/libraries/{library_id}/documents/{document_id}/chunks:bulk", 
                    json=chunk_data
                )
                assert chunks_response.status_code == 201
                
                # Test queries and verify semantic ranking
                test_cases = [
                    {
                        "query": "What are good pets?",
                        "expected_topics": ["dogs", "cats"],  # Should rank pet-related content higher
                        "unexpected_topics": ["machine learning", "cooking"]
                    },
                    {
                        "query": "How does AI work?",
                        "expected_topics": ["machine learning", "deep learning"],
                        "unexpected_topics": ["dogs", "cooking"]
                    },
                    {
                        "query": "What is food preparation?",
                        "expected_topics": ["cooking", "baking"],
                        "unexpected_topics": ["dogs", "machine learning"]
                    }
                ]
                
                for test_case in test_cases:
                    # Generate query embedding
                    query_embedding = generate_cohere_embeddings([test_case["query"]], "search_query")[0]
                    
                    # Search
                    search_data = {
                        "query_embedding": query_embedding,
                        "k": len(topics)
                    }
                    search_response = client.post(f"/libraries/{library_id}/search", json=search_data)
                    assert search_response.status_code == 200
                    search_response_data = search_response.json()
                    search_results = search_response_data["results"]
                    
                    # Get the top result
                    top_result = search_results[0]["text"]
                    
                    # Check that the top result is semantically related
                    found_expected = any(
                        expected in top_result.lower() 
                        for expected in test_case["expected_topics"]
                    )
                    
                    print(f"Query: '{test_case['query']}'")
                    print(f"Top result: '{top_result}'")
                    print(f"Score: {search_results[0]['score']:.4f}")
                    
                    # The top result should be semantically related to the query
                    assert found_expected, f"Top result '{top_result}' should be related to query '{test_case['query']}'"
                
            finally:
                # Clean up
                delete_response = client.delete(f"/libraries/{library_id}")
                assert delete_response.status_code == 204
    
    @pytest.mark.cohere
    def test_batch_embedding_performance(self):
        """Test performance with batch embeddings"""
        # Create a larger batch of texts
        batch_texts = [
            f"This is test text number {i} for batch embedding performance testing."
            for i in range(20)
        ]
        
        start_time = time.time()
        
        # Generate embeddings in batch
        embeddings = generate_cohere_embeddings(batch_texts, "search_document")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Verify all embeddings were generated
        assert len(embeddings) == len(batch_texts)
        
        # Verify all embeddings have correct dimensions
        for embedding in embeddings:
            assert len(embedding) == 1024  # Cohere embed-english-v3.0 dimension
        
        print(f"Generated {len(batch_texts)} embeddings in {processing_time:.2f} seconds")
        print(f"Average time per embedding: {processing_time/len(batch_texts):.3f} seconds")
        
        # Batch processing should be reasonably fast
        assert processing_time < 10.0, f"Batch embedding took too long: {processing_time:.2f} seconds"
