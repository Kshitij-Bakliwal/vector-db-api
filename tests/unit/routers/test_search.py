"""
Unit tests for Search router
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from fastapi.testclient import TestClient
from fastapi import FastAPI

from vector_db_api.api.routers.search import router
from vector_db_api.api.dto import (
    SearchIn, SearchOut, SearchHit, SearchFilters
)
from vector_db_api.services.exceptions import NotFoundError, ValidationError


class TestSearchRouter:
    """Test cases for Search router"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_search_svc = Mock()
        
        # Create FastAPI app with the router
        self.app = FastAPI()
        self.app.include_router(router)
        
        # Add exception handlers
        from vector_db_api.api.errors import register_exception_handlers
        register_exception_handlers(self.app)
        
        # Override the dependency
        def get_mock_search_svc():
            return self.mock_search_svc
        
        # Import the dependency function and override it
        from vector_db_api.api.deps import get_search_svc
        self.app.dependency_overrides[get_search_svc] = get_mock_search_svc
        
        self.client = TestClient(self.app)
        
        self.library_id = uuid4()
        self.chunk_id = uuid4()
        self.document_id = uuid4()
        
        # Sample query embedding (128 dimensions)
        self.query_embedding = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
        
        # Sample search results
        self.sample_hits = [
            {
                "chunk_id": str(self.chunk_id),
                "document_id": str(self.document_id),
                "score": 0.95,
                "text": "This is a relevant chunk of text",
                "position": 0,
                "metadata": {
                    "page_number": 1,
                    "token_count": 15,
                    "author": "Test Author"
                },
                "created_at": "2025-09-10T12:00:00.000000",
                "updated_at": "2025-09-10T12:00:00.000000"
            },
            {
                "chunk_id": str(uuid4()),
                "document_id": str(uuid4()),
                "score": 0.87,
                "text": "Another relevant chunk",
                "position": 1,
                "metadata": {
                    "page_number": 2,
                    "token_count": 12,
                    "author": "Test Author"
                },
                "created_at": "2025-09-10T12:01:00.000000",
                "updated_at": "2025-09-10T12:01:00.000000"
            }
        ]
    
    def test_search_success(self):
        """Test successful search with basic query"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2
        
        # Check first result
        first_result = data["results"][0]
        assert first_result["chunk_id"] == str(self.chunk_id)
        assert first_result["document_id"] == str(self.document_id)
        assert first_result["score"] == 0.95
        assert first_result["text"] == "This is a relevant chunk of text"
        assert first_result["metadata"]["page_number"] == 1
        assert first_result["metadata"]["token_count"] == 15
        assert first_result["metadata"]["author"] == "Test Author"
        
        # Check second result
        second_result = data["results"][1]
        assert second_result["score"] == 0.87
        assert second_result["text"] == "Another relevant chunk"
        
        # Verify service was called correctly
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters=None
        )
    
    def test_search_with_custom_k(self):
        """Test search with custom k value"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits[:1]  # Return only 1 result
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 5
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        
        # Verify service was called with custom k
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=5,
            metric="cosine",
            filters=None
        )
    
    def test_search_with_metric(self):
        """Test search with different similarity metric"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "metric": "euclidean"
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        
        # Verify service was called with metric parameter
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="euclidean",
            filters=None
        )
    
    def test_search_with_filters(self):
        """Test search with filters"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "doc_ids": [str(self.document_id)],
                "author": "Test Author",
                "tags": ["important", "test"]
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        
        # Verify service was called with filters
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={
                "doc_ids": [self.document_id],  # UUID object, not string
                "author": "Test Author",
                "tags": ["important", "test"],
                "created_after": None
            }
        )
    
    def test_search_empty_results(self):
        """Test search with no results"""
        # Arrange
        self.mock_search_svc.query.return_value = []
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 0
        assert data["results"] == []
        
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters=None
        )
    
    def test_search_large_k(self):
        """Test search with large k value"""
        # Arrange
        large_hits = [
            {
                "chunk_id": str(uuid4()),
                "document_id": str(uuid4()),
                "score": 0.9 - (i * 0.01),
                "text": f"Chunk {i}",
                "metadata": {"page_number": i + 1, "token_count": 10 + i},
                "created_at": "2025-09-10T12:00:00.000000",
                "updated_at": "2025-09-10T12:00:00.000000"
            }
            for i in range(50)
        ]
        self.mock_search_svc.query.return_value = large_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 50
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 50
        
        # Verify all results have correct structure
        for i, result in enumerate(data["results"]):
            assert "chunk_id" in result
            assert "document_id" in result
            assert "score" in result
            assert "text" in result
            assert "metadata" in result
            assert result["text"] == f"Chunk {i}"
            assert result["score"] == 0.9 - (i * 0.01)
        
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=50,
            metric="cosine",
            filters=None
        )
    
    def test_search_library_not_found(self):
        """Test search when library doesn't exist"""
        # Arrange
        self.mock_search_svc.query.side_effect = NotFoundError(f"Library with id {self.library_id} not found")
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 404
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters=None
        )
    
    def test_search_validation_error(self):
        """Test search with validation error"""
        # Arrange
        self.mock_search_svc.query.side_effect = ValidationError("Invalid query embedding dimension")
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 422
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters=None
        )
    
    def test_search_missing_query_embedding(self):
        """Test search with missing query embedding"""
        # Arrange
        request_data = {
            "k": 10
            # Missing query_embedding
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from FastAPI
        self.mock_search_svc.query.assert_not_called()
    
    def test_search_invalid_k_value(self):
        """Test search with invalid k value"""
        # Arrange
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 0  # Invalid k value (must be > 0)
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from FastAPI
        self.mock_search_svc.query.assert_not_called()
    
    def test_search_k_too_large(self):
        """Test search with k value too large"""
        # Arrange
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 1001  # Too large (max is 1000)
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from FastAPI
        self.mock_search_svc.query.assert_not_called()
    
    def test_search_invalid_metric(self):
        """Test search with invalid similarity metric"""
        # Arrange
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "metric": "invalid_metric"  # Invalid metric
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from FastAPI
        self.mock_search_svc.query.assert_not_called()
    
    def test_search_invalid_embedding_dimension(self):
        """Test search with invalid embedding dimension"""
        # Arrange
        self.mock_search_svc.query.side_effect = ValidationError("Invalid embedding dimension")
        
        request_data = {
            "query_embedding": [0.1, 0.2, 0.3],  # Wrong dimension (should be 128)
            "k": 10
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 422
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=[0.1, 0.2, 0.3],
            k=10,
            metric="cosine",
            filters=None
        )
    
    def test_search_response_structure(self):
        """Test that search response has correct structure"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check top-level structure
        assert "results" in data
        assert isinstance(data["results"], list)
        
        # Check each result structure
        for result in data["results"]:
            required_fields = ["chunk_id", "document_id", "score", "text", "metadata"]
            for field in required_fields:
                assert field in result
            
            # Check field types
            assert isinstance(result["chunk_id"], str)  # UUID as string
            assert isinstance(result["document_id"], str)  # UUID as string
            assert isinstance(result["score"], (int, float))
            assert isinstance(result["text"], str)
            assert isinstance(result["metadata"], dict)
    
    def test_search_with_complex_metadata(self):
        """Test search with complex metadata in results"""
        # Arrange
        complex_hits = [
            {
                "chunk_id": str(uuid4()),
                "document_id": str(uuid4()),
                "score": 0.95,
                "text": "Complex metadata chunk",
                "position": 0,
                "metadata": {
                    "page_number": 1,
                    "token_count": 20,
                    "author": "Complex Author",
                    "tags": ["important", "complex", "test"],
                    "nested_data": {
                        "section": "introduction",
                        "confidence": 0.9
                    },
                    "numbers": [1, 2, 3, 4, 5]
                },
                "created_at": "2025-09-08T12:00:00.000000",
                "updated_at": "2025-09-08T12:00:00.000000"
            }
        ]
        self.mock_search_svc.query.return_value = complex_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        
        result = data["results"][0]
        metadata = result["metadata"]
        
        # Check complex metadata structure
        assert metadata["page_number"] == 1
        assert metadata["token_count"] == 20
        assert metadata["author"] == "Complex Author"
        assert metadata["tags"] == ["important", "complex", "test"]
        assert metadata["nested_data"]["section"] == "introduction"
        assert metadata["nested_data"]["confidence"] == 0.9
        assert metadata["numbers"] == [1, 2, 3, 4, 5]
    
    def test_search_with_different_metrics(self):
        """Test search with different similarity metrics"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        metrics = ["cosine", "euclidean", "dot_product"]
        
        for metric in metrics:
            # Reset mock for each test
            self.mock_search_svc.reset_mock()
            
            request_data = {
                "query_embedding": self.query_embedding,
                "k": 10,
                "metric": metric
            }
            
            # Act
            response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
            
            # Verify service was called with metric parameter
            self.mock_search_svc.query.assert_called_once_with(
                lib_id=self.library_id,
                query_embedding=self.query_embedding,
                k=10,
                metric=metric,
                filters=None
            )
    
    def test_search_with_empty_filters(self):
        """Test search with empty filters"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {}
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": None, "tags": None, "author": None, "created_after": None}
        )
    
    def test_search_with_partial_filters(self):
        """Test search with partial filters"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "author": "Test Author"
                # Only author filter, no doc_ids or tags
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": None, "tags": None, "author": "Test Author", "created_after": None}
        )
    
    def test_search_score_precision(self):
        """Test search with high precision scores"""
        # Arrange
        precision_hits = [
            {
                "chunk_id": str(uuid4()),
                "document_id": str(uuid4()),
                "score": 0.999999999,  # Very high precision score
                "text": "High precision chunk",
                "metadata": {"page_number": 1, "token_count": 10},
                "created_at": "2025-09-08T12:00:00.000000",
                "updated_at": "2025-09-08T12:00:00.000000"
            },
            {
                "chunk_id": str(uuid4()),
                "document_id": str(uuid4()),
                "score": 0.000000001,  # Very low precision score
                "text": "Low precision chunk",
                "metadata": {"page_number": 2, "token_count": 10},
                "created_at": "2025-09-08T12:00:00.000000",
                "updated_at": "2025-09-08T12:00:00.000000"
            }
        ]
        self.mock_search_svc.query.return_value = precision_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2
        
        # Check that scores are preserved with high precision
        assert data["results"][0]["score"] == 0.999999999
        assert data["results"][1]["score"] == 0.000000001
        
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters=None
        )

    def test_search_with_tag_filter(self):
        """Test search with tag-based filtering"""
        # Arrange
        filtered_hits = [self.sample_hits[0]]  # Only first hit has "test" tag
        self.mock_search_svc.query.return_value = filtered_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "tags": ["test"]
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "This is a relevant chunk of text"
        
        # Verify service was called with tag filter
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": None, "tags": ["test"], "author": None, "created_after": None}
        )

    def test_search_with_multiple_tag_filter(self):
        """Test search with multiple tag filtering"""
        # Arrange
        filtered_hits = [self.sample_hits[0]]  # Only first hit matches
        self.mock_search_svc.query.return_value = filtered_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "tags": ["test", "important"]
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        
        # Verify service was called with multiple tag filter
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": None, "tags": ["test", "important"], "author": None, "created_after": None}
        )

    def test_search_with_author_filter(self):
        """Test search with author-based filtering"""
        # Arrange
        filtered_hits = [self.sample_hits[0]]  # Only first hit has "Test Author"
        self.mock_search_svc.query.return_value = filtered_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "author": "Test Author"
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["metadata"]["author"] == "Test Author"
        
        # Verify service was called with author filter
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": None, "tags": None, "author": "Test Author", "created_after": None}
        )

    def test_search_with_doc_ids_filter(self):
        """Test search with document ID filtering"""
        # Arrange
        filtered_hits = [self.sample_hits[0]]  # Only first hit from specific document
        self.mock_search_svc.query.return_value = filtered_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "doc_ids": [str(self.document_id)]
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["document_id"] == str(self.document_id)
        
        # Verify service was called with doc_ids filter
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": [self.document_id], "tags": None, "author": None, "created_after": None}
        )

    def test_search_with_created_after_filter(self):
        """Test search with created_after date filtering"""
        # Arrange
        from datetime import datetime
        filtered_hits = [self.sample_hits[1]]  # Only second hit is more recent
        self.mock_search_svc.query.return_value = filtered_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "created_after": "2025-09-10T12:00:30.000000"
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        assert data["results"][0]["text"] == "Another relevant chunk"
        
        # Verify service was called with created_after filter
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": None, "tags": None, "author": None, "created_after": datetime(2025, 9, 10, 12, 0, 30)}
        )

    def test_search_with_combined_filters(self):
        """Test search with multiple combined filters"""
        # Arrange
        filtered_hits = [self.sample_hits[0]]  # Only first hit matches all criteria
        self.mock_search_svc.query.return_value = filtered_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "doc_ids": [str(self.document_id)],
                "tags": ["test"],
                "author": "Test Author",
                "created_after": "2025-09-10T12:00:00.000000"
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 1
        result = data["results"][0]
        assert result["document_id"] == str(self.document_id)
        assert result["metadata"]["author"] == "Test Author"
        
        # Verify service was called with combined filters
        from datetime import datetime
        expected_filters = {
            "doc_ids": [self.document_id],  # UUID object, not string
            "tags": ["test"],
            "author": "Test Author",
            "created_after": datetime(2025, 9, 10, 12, 0)  # datetime object, not string
        }
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters=expected_filters
        )

    def test_search_with_no_matching_filters(self):
        """Test search with filters that match no chunks"""
        # Arrange
        self.mock_search_svc.query.return_value = []  # No results match filters
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "tags": ["nonexistent_tag"],
                "author": "Nonexistent Author"
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 0
        
        # Verify service was called with filters
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": None, "tags": ["nonexistent_tag"], "author": "Nonexistent Author", "created_after": None}
        )

    def test_search_with_invalid_date_filter(self):
        """Test search with invalid date format in created_after filter"""
        # Arrange
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "created_after": "invalid-date-format"
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error for invalid date format

    def test_search_with_empty_tag_list_filter(self):
        """Test search with empty tag list filter"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "tags": []
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2  # Should return all results
        
        # Verify service was called with empty tag filter
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": None, "tags": [], "author": None, "created_after": None}
        )

    def test_search_with_empty_doc_ids_filter(self):
        """Test search with empty document ID list filter"""
        # Arrange
        self.mock_search_svc.query.return_value = self.sample_hits
        
        request_data = {
            "query_embedding": self.query_embedding,
            "k": 10,
            "filters": {
                "doc_ids": []
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) == 2  # Should return all results
        
        # Verify service was called with empty doc_ids filter
        self.mock_search_svc.query.assert_called_once_with(
            lib_id=self.library_id,
            query_embedding=self.query_embedding,
            k=10,
            metric="cosine",
            filters={"doc_ids": [], "tags": None, "author": None, "created_after": None}
        )

    def test_search_filters_with_different_metrics(self):
        """Test search filters work with different similarity metrics"""
        # Arrange
        filtered_hits = [self.sample_hits[0]]
        self.mock_search_svc.query.return_value = filtered_hits
        
        metrics = ["cosine", "euclidean", "dot_product"]
        
        for metric in metrics:
            # Reset mock for each test
            self.mock_search_svc.reset_mock()
            
            request_data = {
                "query_embedding": self.query_embedding,
                "k": 10,
                "metric": metric,
                "filters": {
                    "tags": ["test"]
                }
            }
            
            # Act
            response = self.client.post(f"/libraries/{self.library_id}/search", json=request_data)
            
            # Assert
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 1
            
            # Verify service was called with correct metric and filters
            self.mock_search_svc.query.assert_called_once_with(
                lib_id=self.library_id,
                query_embedding=self.query_embedding,
                k=10,
                metric=metric,
                filters={"doc_ids": None, "tags": ["test"], "author": None, "created_after": None}
            )
