"""
Integration tests for FastAPI application
"""

import pytest
import time
import psutil
import os
from fastapi.testclient import TestClient

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from vector_db_api.main import app


class TestAppIntegration:
    """Integration tests for the FastAPI application"""
    
    def test_app_startup_and_shutdown(self):
        """Test that app can start and shutdown cleanly"""
        # Test that we can create a test client
        with TestClient(app) as client:
            # Basic health check
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_app_with_test_client(self):
        """Test app functionality with test client"""
        with TestClient(app) as client:
            # Test multiple endpoints
            health_response = client.get("/health")
            assert health_response.status_code == 200
            
            # Test that we can get OpenAPI schema
            openapi_response = client.get("/openapi.json")
            assert openapi_response.status_code == 200
    
    def test_app_dependency_injection_working(self):
        """Test that dependency injection is working in the app"""
        # Check that the service container is properly set up
        assert hasattr(app.state, 'container')
        container = app.state.container
        assert "library_svc" in container
        
        with TestClient(app) as client:
            # This would test an actual endpoint that uses the service
            # For now, we'll just verify the app can handle requests
            response = client.get("/health")
            assert response.status_code == 200
    
    def test_end_to_end_workflow(self):
        """Test a complete end-to-end workflow"""
        with TestClient(app) as client:
            # 1. Create a library
            library_data = {
                "name": "Test Library",
                "description": "Integration test library",
                "embedding_dim": 128,
                "index_config": {"type": "flat"}
            }
            create_response = client.post("/libraries", json=library_data)
            assert create_response.status_code == 201
            library = create_response.json()
            library_id = library["id"]
            
            # 2. Get the library
            get_response = client.get(f"/libraries/{library_id}")
            assert get_response.status_code == 200
            retrieved_library = get_response.json()
            assert retrieved_library["name"] == "Test Library"
            
            # 3. Create a document
            document_data = {
                "title": "Test Document",
                "content": "This is a test document for integration testing"
            }
            doc_response = client.post(f"/libraries/{library_id}/documents", json=document_data)
            assert doc_response.status_code == 201
            document = doc_response.json()
            document_id = document["id"]
            
            # 4. Create chunks with embeddings
            chunk_data = {
                "chunks": [
                    {
                        "text": "This is the first chunk",
                        "embedding": [0.1] * 128,
                        "position": 0
                    },
                    {
                        "text": "This is the second chunk", 
                        "embedding": [0.2] * 128,
                        "position": 1
                    }
                ]
            }
            chunks_response = client.post(f"/libraries/{library_id}/documents/{document_id}/chunks:bulk", json=chunk_data)
            assert chunks_response.status_code == 201
            chunks_result = chunks_response.json()
            assert "chunk_ids" in chunks_result
            assert len(chunks_result["chunk_ids"]) == 2
            
            # 5. Search for similar chunks
            search_data = {
                "query_embedding": [0.15] * 128,
                "limit": 2
            }
            search_response = client.post(f"/libraries/{library_id}/search", json=search_data)
            assert search_response.status_code == 200
            search_results = search_response.json()
            assert len(search_results) >= 1
            
            # 6. Clean up - delete the library (which should cascade delete documents and chunks)
            delete_response = client.delete(f"/libraries/{library_id}")
            assert delete_response.status_code == 204
    
    def test_concurrent_requests(self):
        """Test that the app can handle concurrent requests"""
        with TestClient(app) as client:
            import threading
            import time
            
            results = []
            
            def make_request():
                response = client.get("/health")
                results.append(response.status_code)
            
            # Create multiple threads making concurrent requests
            threads = []
            for _ in range(10):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # All requests should succeed
            assert len(results) == 10
            assert all(status == 200 for status in results)
    
    def test_error_handling_integration(self):
        """Test error handling in an integrated scenario"""
        with TestClient(app) as client:
            # Test 404 for non-existent library
            response = client.get("/libraries/00000000-0000-0000-0000-000000000000")
            assert response.status_code == 404
            
            # Test 404 for non-existent document
            response = client.get("/libraries/00000000-0000-0000-0000-000000000000/documents/00000000-0000-0000-0000-000000000000")
            assert response.status_code == 404
            
            # Test validation error for invalid data
            invalid_library_data = {
                "name": "",  # Empty name should fail validation
                "embedding_dim": -1  # Negative dimension should fail validation
            }
            response = client.post("/libraries", json=invalid_library_data)
            assert response.status_code == 422  # Validation error


class TestAppPerformance:
    """Performance tests for the FastAPI application"""
    
    def test_app_startup_time(self):
        """Test that app starts up quickly"""
        start_time = time.time()
        
        # Create a new test client (this initializes the app)
        with TestClient(app) as client:
            pass
        
        end_time = time.time()
        startup_time = end_time - start_time
        
        # App should start up in less than 1 second
        assert startup_time < 1.0, f"App startup took {startup_time:.2f} seconds"
    
    def test_app_memory_usage(self):
        """Test that app doesn't use excessive memory"""
        # Get current process memory usage
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Create test client and make requests
        with TestClient(app) as client:
            # Make a few requests
            for _ in range(10):
                client.get("/health")
        
        memory_after = process.memory_info().rss
        memory_increase = memory_after - memory_before
        
        # Memory increase should be reasonable (less than 50MB)
        assert memory_increase < 50 * 1024 * 1024, f"Memory increased by {memory_increase / 1024 / 1024:.2f}MB"
    
    def test_response_times(self):
        """Test that API responses are fast enough"""
        with TestClient(app) as client:
            # Test health endpoint response time
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response.status_code == 200
            assert response_time < 0.1, f"Health endpoint took {response_time:.3f} seconds"
            
            # Test OpenAPI schema response time
            start_time = time.time()
            response = client.get("/openapi.json")
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response.status_code == 200
            assert response_time < 0.5, f"OpenAPI schema took {response_time:.3f} seconds"
