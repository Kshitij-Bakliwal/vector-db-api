"""
Unit tests for Chunks router
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

from vector_db_api.api.routers.chunks import router
from vector_db_api.api.dto import (
    ChunkIn, ChunkOut, BulkChunksIn, BulkChunksOut
)
from vector_db_api.models.entities import Chunk
from vector_db_api.models.metadata import ChunkMetadata
from vector_db_api.services.exceptions import NotFoundError, ValidationError, ConflictError


class TestChunksRouter:
    """Test cases for Chunks router"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_chunk_svc = Mock()
        
        # Create FastAPI app with the router
        self.app = FastAPI()
        self.app.include_router(router)
        
        # Add exception handlers
        from vector_db_api.api.errors import register_exception_handlers
        register_exception_handlers(self.app)
        
        # Override the dependency
        def get_mock_chunk_svc():
            return self.mock_chunk_svc
        
        # Import the dependency function and override it
        from vector_db_api.api.deps import get_chunk_svc
        self.app.dependency_overrides[get_chunk_svc] = get_mock_chunk_svc
        
        self.client = TestClient(self.app)
        
        self.library_id = uuid4()
        self.document_id = uuid4()
        self.chunk_id = uuid4()
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        self.mock_chunk = Chunk(
            id=self.chunk_id,
            library_id=self.library_id,
            document_id=self.document_id,
            text="Test chunk text",
            position=0,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2],  # 128 dimensions
            metadata=ChunkMetadata(page_number=1, token_count=10),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
    
    def test_upsert_chunk_success(self):
        """Test successful chunk upsert"""
        # Arrange
        self.mock_chunk_svc.upsert.return_value = self.mock_chunk
        
        request_data = {
            "text": "Test chunk text",
            "position": 0,
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            "metadata": {
                "page_number": 1,
                "token_count": 10
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(self.chunk_id)
        assert data["library_id"] == str(self.library_id)
        assert data["document_id"] == str(self.document_id)
        assert data["text"] == "Test chunk text"
        assert data["position"] == 0
        assert data["metadata"]["page_number"] == 1
        assert data["metadata"]["token_count"] == 10
        assert data["version"] == 1
        
        self.mock_chunk_svc.upsert.assert_called_once()
        call_args = self.mock_chunk_svc.upsert.call_args
        chunk = call_args[0][0]
        assert chunk.text == "Test chunk text"
        assert chunk.position == 0
        assert chunk.library_id == self.library_id
        assert chunk.document_id == self.document_id
    
    def test_upsert_chunk_with_id(self):
        """Test chunk upsert with provided ID"""
        # Arrange
        self.mock_chunk_svc.upsert.return_value = self.mock_chunk
        
        request_data = {
            "id": str(self.chunk_id),
            "text": "Test chunk text",
            "position": 0,
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            "metadata": {}
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(self.chunk_id)
        
        self.mock_chunk_svc.upsert.assert_called_once()
        call_args = self.mock_chunk_svc.upsert.call_args
        chunk = call_args[0][0]
        assert chunk.id == self.chunk_id
    
    def test_upsert_chunk_minimal_data(self):
        """Test chunk upsert with minimal required data"""
        # Arrange
        minimal_chunk = Chunk(
            id=self.chunk_id,
            library_id=self.library_id,
            document_id=self.document_id,
            text="Minimal chunk",
            position=0,
            embedding=None,
            metadata=ChunkMetadata(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        self.mock_chunk_svc.upsert.return_value = minimal_chunk
        
        request_data = {
            "text": "Minimal chunk"
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "Minimal chunk"
        assert data["position"] == 0
        
        self.mock_chunk_svc.upsert.assert_called_once()
    
    def test_upsert_chunk_validation_error(self):
        """Test chunk upsert with validation error"""
        # Arrange
        self.mock_chunk_svc.upsert.side_effect = ValidationError("Invalid embedding dimension")
        
        request_data = {
            "text": "Test chunk",
            "embedding": [0.1, 0.2, 0.3]  # Wrong dimension
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 422
        self.mock_chunk_svc.upsert.assert_called_once()
    
    def test_upsert_chunk_not_found(self):
        """Test chunk upsert when library or document doesn't exist"""
        # Arrange
        self.mock_chunk_svc.upsert.side_effect = NotFoundError(f"Library with id {self.library_id} not found")
        
        request_data = {
            "text": "Test chunk",
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 404
        self.mock_chunk_svc.upsert.assert_called_once()
    
    def test_upsert_chunk_missing_text(self):
        """Test chunk upsert with missing required text field"""
        # Arrange
        request_data = {
            "position": 0,
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
            # Missing text field
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from FastAPI
        self.mock_chunk_svc.upsert.assert_not_called()
    
    def test_upsert_chunk_invalid_position(self):
        """Test chunk upsert with invalid position"""
        # Arrange
        request_data = {
            "text": "Test chunk",
            "position": -1,  # Invalid position
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from FastAPI
        self.mock_chunk_svc.upsert.assert_not_called()
    
    def test_bulk_upsert_chunks_success(self):
        """Test successful bulk chunk upsert"""
        # Arrange
        chunk1 = Chunk(
            id=uuid4(),
            library_id=self.library_id,
            document_id=self.document_id,
            text="First chunk",
            position=0,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            metadata=ChunkMetadata(page_number=1, token_count=10),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        chunk2 = Chunk(
            id=uuid4(),
            library_id=self.library_id,
            document_id=self.document_id,
            text="Second chunk",
            position=1,
            embedding=[0.4, 0.5, 0.6] * 42 + [0.4, 0.5],
            metadata=ChunkMetadata(page_number=2, token_count=15),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        
        self.mock_chunk_svc.bulk_upsert.return_value = [chunk1, chunk2]
        
        request_data = {
            "chunks": [
                {
                    "text": "First chunk",
                    "position": 0,
                    "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
                    "metadata": {"page_number": 1, "token_count": 10}
                },
                {
                    "text": "Second chunk",
                    "position": 1,
                    "embedding": [0.4, 0.5, 0.6] * 42 + [0.4, 0.5],
                    "metadata": {"page_number": 2, "token_count": 15}
                }
            ]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks:bulk", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert len(data["chunk_ids"]) == 2
        assert data["chunk_ids"][0] == str(chunk1.id)
        assert data["chunk_ids"][1] == str(chunk2.id)
        
        self.mock_chunk_svc.bulk_upsert.assert_called_once()
        call_args = self.mock_chunk_svc.bulk_upsert.call_args
        assert call_args[0][0] == self.library_id  # lib_id
        assert call_args[0][1] == self.document_id  # doc_id
        chunks = call_args[0][2]
        assert len(chunks) == 2
        assert chunks[0].text == "First chunk"
        assert chunks[1].text == "Second chunk"
    
    def test_bulk_upsert_chunks_single_chunk(self):
        """Test bulk upsert with single chunk"""
        # Arrange
        self.mock_chunk_svc.bulk_upsert.return_value = [self.mock_chunk]
        
        request_data = {
            "chunks": [
                {
                    "text": "Single chunk",
                    "position": 0,
                    "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
                    "metadata": {}
                }
            ]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks:bulk", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert len(data["chunk_ids"]) == 1
        assert data["chunk_ids"][0] == str(self.chunk_id)
        
        self.mock_chunk_svc.bulk_upsert.assert_called_once()
    
    def test_bulk_upsert_chunks_validation_error(self):
        """Test bulk upsert with validation error"""
        # Arrange
        self.mock_chunk_svc.bulk_upsert.side_effect = ValidationError("Invalid chunk data")
        
        request_data = {
            "chunks": [
                {
                    "text": "Test chunk",
                    "position": 0,
                    "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
                    "metadata": {}
                }
            ]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks:bulk", json=request_data)
        
        # Assert
        assert response.status_code == 422
        self.mock_chunk_svc.bulk_upsert.assert_called_once()
    
    def test_bulk_upsert_chunks_empty_list(self):
        """Test bulk upsert with empty chunks list"""
        # Arrange
        request_data = {
            "chunks": []  # Empty chunks list
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks:bulk", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from FastAPI
        self.mock_chunk_svc.bulk_upsert.assert_not_called()
    
    def test_bulk_upsert_chunks_large_batch(self):
        """Test bulk upsert with large batch of chunks"""
        # Arrange
        chunks = []
        for i in range(10):
            chunk = Chunk(
                id=uuid4(),
                library_id=self.library_id,
                document_id=self.document_id,
                text=f"Chunk {i}",
                position=i,
                embedding=[0.1 * i, 0.2 * i, 0.3 * i] * 42 + [0.1 * i, 0.2 * i],
                metadata=ChunkMetadata(page_number=i + 1, token_count=10 + i),
                created_at=self.created_at,
                updated_at=self.updated_at,
                version=1
            )
            chunks.append(chunk)
        
        self.mock_chunk_svc.bulk_upsert.return_value = chunks
        
        request_data = {
            "chunks": [
                {
                    "text": f"Chunk {i}",
                    "position": i,
                    "embedding": [0.1 * i, 0.2 * i, 0.3 * i] * 42 + [0.1 * i, 0.2 * i],
                    "metadata": {"page_number": i + 1, "token_count": 10 + i}
                }
                for i in range(10)
            ]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks:bulk", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert len(data["chunk_ids"]) == 10
        
        self.mock_chunk_svc.bulk_upsert.assert_called_once()
        call_args = self.mock_chunk_svc.bulk_upsert.call_args
        chunks = call_args[0][2]
        assert len(chunks) == 10
        for i, chunk in enumerate(chunks):
            assert chunk.text == f"Chunk {i}"
            assert chunk.position == i
    
    def test_delete_chunk_success(self):
        """Test successful chunk deletion"""
        # Arrange
        self.mock_chunk_svc.delete.return_value = None
        
        # Act
        response = self.client.delete(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks/{self.chunk_id}")
        
        # Assert
        assert response.status_code == 204
        assert response.content == b""  # No content for 204
        self.mock_chunk_svc.delete.assert_called_once_with(self.library_id, self.chunk_id)
    
    def test_delete_chunk_not_found(self):
        """Test chunk deletion when chunk doesn't exist"""
        # Arrange
        self.mock_chunk_svc.delete.side_effect = NotFoundError(f"Chunk with id {self.chunk_id} not found")
        
        # Act
        response = self.client.delete(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks/{self.chunk_id}")
        
        # Assert
        assert response.status_code == 404
        self.mock_chunk_svc.delete.assert_called_once_with(self.library_id, self.chunk_id)
    
    def test_delete_chunk_invalid_uuid(self):
        """Test chunk deletion with invalid UUID"""
        # Act
        response = self.client.delete(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks/invalid-uuid")
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_chunk_svc.delete.assert_not_called()
    
    def test_chunk_response_structure(self):
        """Test that chunk response has correct structure"""
        # Arrange
        self.mock_chunk_svc.upsert.return_value = self.mock_chunk
        
        request_data = {
            "text": "Test chunk",
            "position": 0,
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            "metadata": {"page_number": 1, "token_count": 10}
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields are present
        required_fields = [
            "id", "library_id", "document_id", "text", "position",
            "metadata", "created_at", "updated_at", "version"
        ]
        for field in required_fields:
            assert field in data
        
        # Check field types
        assert isinstance(data["id"], str)  # UUID as string
        assert isinstance(data["library_id"], str)  # UUID as string
        assert isinstance(data["document_id"], str)  # UUID as string
        assert isinstance(data["text"], str)
        assert isinstance(data["position"], int)
        assert isinstance(data["metadata"], dict)
        assert isinstance(data["version"], int)
    
    def test_chunk_request_to_entity_conversion(self):
        """Test that chunk request data is properly converted to entity"""
        # Arrange
        self.mock_chunk_svc.upsert.return_value = self.mock_chunk
        
        request_data = {
            "text": "Test chunk conversion",
            "position": 5,
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            "metadata": {"page_number": 3, "token_count": 20}
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 200
        
        # Verify the chunk was properly converted
        call_args = self.mock_chunk_svc.upsert.call_args
        chunk = call_args[0][0]
        assert chunk.text == "Test chunk conversion"
        assert chunk.position == 5
        assert chunk.library_id == self.library_id
        assert chunk.document_id == self.document_id
        assert chunk.embedding == [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
        # Note: metadata is passed as dict, not converted to ChunkMetadata object
    
    def test_chunk_with_standard_metadata(self):
        """Test chunk creation with standard metadata fields"""
        # Arrange
        self.mock_chunk_svc.upsert.return_value = self.mock_chunk
        
        request_data = {
            "text": "Chunk with standard metadata",
            "position": 0,
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            "metadata": {
                "page_number": 1,
                "token_count": 10,
                "author": "Test Author",
                "tags": ["test", "chunk"]
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks", json=request_data)
        
        # Assert
        assert response.status_code == 200
        
        # Verify the standard metadata was preserved
        call_args = self.mock_chunk_svc.upsert.call_args
        chunk = call_args[0][0]
        assert chunk.metadata.page_number == 1
        assert chunk.metadata.token_count == 10
        assert chunk.metadata.author == "Test Author"
        assert chunk.metadata.tags == ["test", "chunk"]
    
    def test_bulk_upsert_with_mixed_chunk_data(self):
        """Test bulk upsert with chunks having different data patterns"""
        # Arrange
        chunk1 = Chunk(
            id=uuid4(),
            library_id=self.library_id,
            document_id=self.document_id,
            text="Chunk with embedding",
            position=0,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            metadata=ChunkMetadata(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        chunk2 = Chunk(
            id=uuid4(),
            library_id=self.library_id,
            document_id=self.document_id,
            text="Chunk without embedding",
            position=1,
            embedding=None,
            metadata=ChunkMetadata(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        
        self.mock_chunk_svc.bulk_upsert.return_value = [chunk1, chunk2]
        
        request_data = {
            "chunks": [
                {
                    "text": "Chunk with embedding",
                    "position": 0,
                    "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
                    "metadata": {}
                },
                {
                    "text": "Chunk without embedding",
                    "position": 1,
                    "metadata": {}
                }
            ]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}/chunks:bulk", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert len(data["chunk_ids"]) == 2
        
        # Verify both chunks were properly converted
        call_args = self.mock_chunk_svc.bulk_upsert.call_args
        chunks = call_args[0][2]
        assert len(chunks) == 2
        assert chunks[0].text == "Chunk with embedding"
        assert chunks[0].embedding is not None
        assert chunks[1].text == "Chunk without embedding"
        assert chunks[1].embedding is None
