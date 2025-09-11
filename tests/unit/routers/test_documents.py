"""
Unit tests for Documents router
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

from vector_db_api.api.routers.documents import router
from vector_db_api.api.dto import (
    DocumentCreate, DocumentCreateWithChunks, DocumentOut,
    Page, PageMetadata, ChunkIn, MoveDocumentIn
)
from vector_db_api.models.entities import Document, Chunk
from vector_db_api.models.metadata import DocumentMetadata, ChunkMetadata
from vector_db_api.services.exceptions import NotFoundError, ValidationError, ConflictError


class TestDocumentsRouter:
    """Test cases for Documents router"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_document_svc = Mock()
        
        # Create FastAPI app with the router
        self.app = FastAPI()
        self.app.include_router(router)
        
        # Add exception handlers
        from vector_db_api.api.errors import register_exception_handlers
        register_exception_handlers(self.app)
        
        # Override the dependency
        def get_mock_document_svc():
            return self.mock_document_svc
        
        # Import the dependency function and override it
        from vector_db_api.api.deps import get_document_svc
        self.app.dependency_overrides[get_document_svc] = get_mock_document_svc
        
        self.client = TestClient(self.app)
        
        self.library_id = uuid4()
        self.document_id = uuid4()
        self.chunk_id = uuid4()
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        self.mock_document = Document(
            id=self.document_id,
            library_id=self.library_id,
            metadata=DocumentMetadata(title="Test Document", summary="Test document summary"),
            chunk_ids=[self.chunk_id],
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        
        self.mock_chunk = Chunk(
            id=self.chunk_id,
            library_id=self.library_id,
            document_id=self.document_id,
            text="Test chunk text",
            position=0,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2],  # 128 dimensions
            metadata=ChunkMetadata(page_number=1, token_count=10)
        )
    
    def test_create_document_success(self):
        """Test successful document creation"""
        # Arrange
        self.mock_document_svc.create.return_value = self.mock_document
        
        request_data = {
            "metadata": {
                "title": "Test Document",
                "summary": "Test document summary"
            }
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == str(self.document_id)
        assert data["library_id"] == str(self.library_id)
        assert data["metadata"]["title"] == "Test Document"
        assert data["metadata"]["summary"] == "Test document summary"
        assert data["chunk_ids"] == [str(self.chunk_id)]
        assert data["version"] == 1
        
        self.mock_document_svc.create.assert_called_once()
        call_args = self.mock_document_svc.create.call_args
        assert call_args.kwargs["lib_id"] == self.library_id
        assert call_args.kwargs["metadata"]["title"] == "Test Document"
    
    def test_create_document_empty_metadata(self):
        """Test document creation with empty metadata"""
        # Arrange
        doc_empty_metadata = Document(
            id=self.document_id,
            library_id=self.library_id,
            metadata=DocumentMetadata(),
            chunk_ids=[],
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        self.mock_document_svc.create.return_value = doc_empty_metadata
        
        request_data = {"metadata": {}}
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == str(self.document_id)
        # Check that metadata is a dict with default values (not empty)
        assert isinstance(data["metadata"], dict)
        assert "title" in data["metadata"]
        assert "summary" in data["metadata"]
        assert data["chunk_ids"] == []
        
        self.mock_document_svc.create.assert_called_once()
    
    def test_create_document_validation_error(self):
        """Test document creation with validation error from service"""
        # Arrange
        self.mock_document_svc.create.side_effect = ValidationError("Invalid metadata")
        
        request_data = {"metadata": {"title": "Test Document"}}
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents", json=request_data)
        
        # Assert
        assert response.status_code == 422
        self.mock_document_svc.create.assert_called_once()
    
    def test_create_document_not_found(self):
        """Test document creation when library doesn't exist"""
        # Arrange
        self.mock_document_svc.create.side_effect = NotFoundError(f"Library with id {self.library_id} not found")
        
        request_data = {"metadata": {"title": "Test Document"}}
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents", json=request_data)
        
        # Assert
        assert response.status_code == 404
        self.mock_document_svc.create.assert_called_once()
    
    def test_create_document_with_chunks_success(self):
        """Test successful document creation with chunks"""
        # Arrange
        self.mock_document_svc.create.return_value = self.mock_document
        self.mock_document_svc.create_with_chunks.return_value = self.mock_document
        
        request_data = {
            "metadata": {
                "title": "Test Document",
                "summary": "Test document summary"
            },
            "chunks": [
                {
                    "text": "First chunk text",
                    "position": 0,
                    "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
                    "metadata": {"page_number": 1, "token_count": 10}
                },
                {
                    "text": "Second chunk text",
                    "position": 1,
                    "embedding": [0.4, 0.5, 0.6] * 42 + [0.4, 0.5],
                    "metadata": {"page_number": 2, "token_count": 15}
                }
            ]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/with-chunks", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == str(self.document_id)
        assert data["library_id"] == str(self.library_id)
        assert data["metadata"]["title"] == "Test Document"
        assert data["chunk_ids"] == [str(self.chunk_id)]
        
        # Verify both service methods were called
        self.mock_document_svc.create.assert_called_once()
        self.mock_document_svc.create_with_chunks.assert_called_once()
        
        # Verify create_with_chunks was called with correct parameters
        call_args = self.mock_document_svc.create_with_chunks.call_args
        assert call_args.kwargs["lib_id"] == self.library_id
        assert len(call_args.kwargs["chunks"]) == 2
        assert call_args.kwargs["chunks"][0].text == "First chunk text"
        assert call_args.kwargs["chunks"][1].text == "Second chunk text"
    
    def test_create_document_with_chunks_validation_error(self):
        """Test document creation with chunks when validation fails"""
        # Arrange
        self.mock_document_svc.create.return_value = self.mock_document
        self.mock_document_svc.create_with_chunks.side_effect = ValidationError("Invalid chunk data")
        
        request_data = {
            "metadata": {"title": "Test Document"},
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
        response = self.client.post(f"/libraries/{self.library_id}/documents/with-chunks", json=request_data)
        
        # Assert
        assert response.status_code == 422
        self.mock_document_svc.create.assert_called_once()
        self.mock_document_svc.create_with_chunks.assert_called_once()
    
    def test_create_document_with_chunks_missing_chunks(self):
        """Test document creation with chunks when chunks list is empty"""
        # Arrange
        request_data = {
            "metadata": {"title": "Test Document"},
            "chunks": []  # Empty chunks list
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/with-chunks", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from FastAPI
        self.mock_document_svc.create.assert_not_called()
        self.mock_document_svc.create_with_chunks.assert_not_called()
    
    def test_list_documents_success(self):
        """Test successful document listing"""
        # Arrange
        doc1 = Document(
            id=uuid4(),
            library_id=self.library_id,
            metadata=DocumentMetadata(title="Document 1"),
            chunk_ids=[],
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        doc2 = Document(
            id=uuid4(),
            library_id=self.library_id,
            metadata=DocumentMetadata(title="Document 2"),
            chunk_ids=[],
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        
        self.mock_document_svc.list.return_value = [doc1, doc2]
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}/documents?limit=10&offset=0")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["page"]["limit"] == 10
        assert data["page"]["offset"] == 0
        assert data["page"]["has_more"] is False
        
        assert data["items"][0]["metadata"]["title"] == "Document 1"
        assert data["items"][1]["metadata"]["title"] == "Document 2"
        
        assert self.mock_document_svc.list.call_count == 1
    
    def test_list_documents_with_pagination(self):
        """Test document listing with pagination"""
        # Arrange
        docs = []
        for i in range(25):
            doc = Document(
                id=uuid4(),
                library_id=self.library_id,
                metadata=DocumentMetadata(title=f"Document {i}"),
                chunk_ids=[],
                created_at=self.created_at,
                updated_at=self.updated_at,
                version=1
            )
            docs.append(doc)
        
        # Return 11 items to simulate has_more = True
        self.mock_document_svc.list.return_value = docs[10:21]  # 11 items (10 + 1 extra)
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}/documents?limit=10&offset=10")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 10  # Should be trimmed to limit
        assert data["page"]["limit"] == 10
        assert data["page"]["offset"] == 10
        assert data["page"]["has_more"] is True
        
        assert self.mock_document_svc.list.call_count == 1
    
    def test_list_documents_default_params(self):
        """Test document listing with default parameters"""
        # Arrange
        self.mock_document_svc.list.return_value = [self.mock_document]
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}/documents")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["page"]["limit"] == 50  # Default limit
        assert data["page"]["offset"] == 0   # Default offset
        assert self.mock_document_svc.list.call_count == 1
    
    def test_list_documents_validation_error(self):
        """Test document listing with invalid parameters"""
        # Act
        response = self.client.get(f"/libraries/{self.library_id}/documents?limit=2000&offset=-1")
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_document_svc.list.assert_not_called()
    
    def test_get_document_success(self):
        """Test successful document retrieval"""
        # Arrange
        self.mock_document_svc.get.return_value = self.mock_document
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}/documents/{self.document_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(self.document_id)
        assert data["library_id"] == str(self.library_id)
        assert data["metadata"]["title"] == "Test Document"
        assert data["chunk_ids"] == [str(self.chunk_id)]
        assert data["version"] == 1
        
        self.mock_document_svc.get.assert_called_once_with(self.library_id, self.document_id)
    
    def test_get_document_not_found(self):
        """Test document retrieval when document doesn't exist"""
        # Arrange
        self.mock_document_svc.get.side_effect = NotFoundError(f"Document with id {self.document_id} not found")
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}/documents/{self.document_id}")
        
        # Assert
        assert response.status_code == 404
        self.mock_document_svc.get.assert_called_once_with(self.library_id, self.document_id)
    
    def test_get_document_invalid_uuid(self):
        """Test document retrieval with invalid UUID"""
        # Act
        response = self.client.get(f"/libraries/{self.library_id}/documents/invalid-uuid")
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_document_svc.get.assert_not_called()
    
    def test_move_document_success(self):
        """Test successful document move"""
        # Arrange
        dst_library_id = uuid4()
        moved_document = Document(
            id=self.document_id,
            library_id=dst_library_id,
            metadata=DocumentMetadata(title="Test Document", summary="Test document summary"),
            chunk_ids=[self.chunk_id],
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=2
        )
        self.mock_document_svc.move_to_library.return_value = moved_document
        
        request_data = {"dst_library_id": str(dst_library_id)}
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}:move", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(self.document_id)
        assert data["library_id"] == str(dst_library_id)
        assert data["version"] == 2
        
        self.mock_document_svc.move_to_library.assert_called_once_with(
            self.document_id, self.library_id, dst_library_id
        )
    
    def test_move_document_not_found(self):
        """Test document move when document doesn't exist"""
        # Arrange
        dst_library_id = uuid4()
        self.mock_document_svc.move_to_library.side_effect = NotFoundError(f"Document with id {self.document_id} not found")
        
        request_data = {"dst_library_id": str(dst_library_id)}
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}:move", json=request_data)
        
        # Assert
        assert response.status_code == 404
        self.mock_document_svc.move_to_library.assert_called_once()
    
    def test_move_document_validation_error(self):
        """Test document move with validation error"""
        # Arrange
        dst_library_id = uuid4()
        self.mock_document_svc.move_to_library.side_effect = ValidationError("Invalid destination library")
        
        request_data = {"dst_library_id": str(dst_library_id)}
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}:move", json=request_data)
        
        # Assert
        assert response.status_code == 422
        self.mock_document_svc.move_to_library.assert_called_once()
    
    def test_move_document_missing_dst_library_id(self):
        """Test document move with missing destination library ID"""
        # Arrange
        request_data = {}  # Missing dst_library_id
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/{self.document_id}:move", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_document_svc.move_to_library.assert_not_called()
    
    def test_delete_document_success(self):
        """Test successful document deletion"""
        # Arrange
        self.mock_document_svc.delete.return_value = None
        
        # Act
        response = self.client.delete(f"/libraries/{self.library_id}/documents/{self.document_id}")
        
        # Assert
        assert response.status_code == 204
        assert response.content == b""  # No content for 204
        self.mock_document_svc.delete.assert_called_once_with(self.library_id, self.document_id)
    
    def test_delete_document_not_found(self):
        """Test document deletion when document doesn't exist"""
        # Arrange
        self.mock_document_svc.delete.side_effect = NotFoundError(f"Document with id {self.document_id} not found")
        
        # Act
        response = self.client.delete(f"/libraries/{self.library_id}/documents/{self.document_id}")
        
        # Assert
        assert response.status_code == 404
        self.mock_document_svc.delete.assert_called_once_with(self.library_id, self.document_id)
    
    def test_delete_document_invalid_uuid(self):
        """Test document deletion with invalid UUID"""
        # Act
        response = self.client.delete(f"/libraries/{self.library_id}/documents/invalid-uuid")
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_document_svc.delete.assert_not_called()
    
    def test_document_response_structure(self):
        """Test that document response has correct structure"""
        # Arrange
        self.mock_document_svc.get.return_value = self.mock_document
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}/documents/{self.document_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields are present
        required_fields = [
            "id", "library_id", "metadata", "chunk_ids", 
            "created_at", "updated_at", "version"
        ]
        for field in required_fields:
            assert field in data
        
        # Check field types
        assert isinstance(data["id"], str)  # UUID as string
        assert isinstance(data["library_id"], str)  # UUID as string
        assert isinstance(data["metadata"], dict)
        assert isinstance(data["chunk_ids"], list)
        assert isinstance(data["version"], int)
    
    def test_chunk_request_to_entity_conversion(self):
        """Test that chunk request data is properly converted to entity"""
        # Arrange
        self.mock_document_svc.create.return_value = self.mock_document
        self.mock_document_svc.create_with_chunks.return_value = self.mock_document
        
        request_data = {
            "metadata": {"title": "Test Document"},
            "chunks": [
                {
                    "text": "Test chunk",
                    "position": 5,
                    "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
                    "metadata": {"page_number": 3, "token_count": 20}
                }
            ]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/with-chunks", json=request_data)
        
        # Assert
        assert response.status_code == 201
        
        # Verify the chunk was properly converted
        call_args = self.mock_document_svc.create_with_chunks.call_args
        chunk = call_args.kwargs["chunks"][0]
        assert chunk.text == "Test chunk"
        assert chunk.position == 5
        assert chunk.library_id == self.library_id
        assert chunk.document_id == self.mock_document.id
        assert chunk.metadata.page_number == 3
        assert chunk.metadata.token_count == 20
    
    def test_document_with_multiple_chunks(self):
        """Test document creation with multiple chunks"""
        # Arrange
        self.mock_document_svc.create.return_value = self.mock_document
        self.mock_document_svc.create_with_chunks.return_value = self.mock_document
        
        request_data = {
            "metadata": {"title": "Multi-chunk Document"},
            "chunks": [
                {
                    "text": f"Chunk {i}",
                    "position": i,
                    "embedding": [0.1 * i, 0.2 * i, 0.3 * i] * 42 + [0.1 * i, 0.2 * i],
                    "metadata": {"page_number": i + 1, "token_count": 10 + i}
                }
                for i in range(5)
            ]
        }
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/documents/with-chunks", json=request_data)
        
        # Assert
        assert response.status_code == 201
        
        # Verify all chunks were properly converted
        call_args = self.mock_document_svc.create_with_chunks.call_args
        chunks = call_args.kwargs["chunks"]
        assert len(chunks) == 5
        
        for i, chunk in enumerate(chunks):
            assert chunk.text == f"Chunk {i}"
            assert chunk.position == i
            assert chunk.metadata.page_number == i + 1
            assert chunk.metadata.token_count == 10 + i
