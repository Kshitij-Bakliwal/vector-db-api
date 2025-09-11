"""
Unit tests for Libraries router
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

from vector_db_api.api.routers.libraries import router
from vector_db_api.api.dto import (
    LibraryCreate, LibraryOut, IndexConfigIn, IndexConfigOut,
    Page, PageMetadata, RebuildIndexOut
)
from vector_db_api.models.entities import Library
from vector_db_api.models.metadata import LibraryMetadata
from vector_db_api.models.indexing import IndexType
from vector_db_api.services.exceptions import NotFoundError, ValidationError, ConflictError


class TestLibrariesRouter:
    """Test cases for Libraries router"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_library_svc = Mock()
        
        # Create FastAPI app with the router
        self.app = FastAPI()
        self.app.include_router(router)
        
        # Add exception handlers
        from vector_db_api.api.errors import register_exception_handlers
        register_exception_handlers(self.app)
        
        # Override the dependency
        def get_mock_library_svc():
            return self.mock_library_svc
        
        # Import the dependency function and override it
        from vector_db_api.api.deps import get_library_svc
        self.app.dependency_overrides[get_library_svc] = get_mock_library_svc
        
        self.client = TestClient(self.app)
        
        self.library_id = uuid4()
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        
        self.mock_library = Library(
            id=self.library_id,
            name="Test Library",
            embedding_dim=128,
            index_config=IndexType(type="flat"),
            metadata=LibraryMetadata(description="Test library description"),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
    
    def test_create_library_success(self):
        """Test successful library creation"""
        # Arrange
        self.mock_library_svc.create.return_value = self.mock_library
        
        request_data = {
            "name": "Test Library",
            "embedding_dim": 128,
            "index_config": {
                "type": "flat"
            },
            "metadata": {
                "description": "Test library description"
            }
        }
        
        # Act
        response = self.client.post("/libraries", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == str(self.library_id)
        assert data["name"] == "Test Library"
        assert data["embedding_dim"] == 128
        assert data["index_config"]["type"] == "flat"
        assert data["metadata"]["description"] == "Test library description"
        assert data["version"] == 1
        
        self.mock_library_svc.create.assert_called_once()
        call_args = self.mock_library_svc.create.call_args
        assert call_args.kwargs["name"] == "Test Library"
        assert call_args.kwargs["embedding_dim"] == 128
        assert call_args.kwargs["index_config"].type == "flat"
    
    def test_create_library_with_lsh_config(self):
        """Test library creation with LSH index configuration"""
        # Arrange
        lsh_library = Library(
            id=self.library_id,
            name="LSH Library",
            embedding_dim=256,
            index_config=IndexType(type="lsh"),
            metadata=LibraryMetadata(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        self.mock_library_svc.create.return_value = lsh_library
        
        request_data = {
            "name": "LSH Library",
            "embedding_dim": 256,
            "index_config": {
                "type": "lsh",
                "lsh": {
                    "num_tables": 16,
                    "hyperplanes_per_table": 32
                }
            },
            "metadata": {}
        }
        
        # Act
        response = self.client.post("/libraries", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "LSH Library"
        assert data["embedding_dim"] == 256
        assert data["index_config"]["type"] == "lsh"
        
        self.mock_library_svc.create.assert_called_once()
        call_args = self.mock_library_svc.create.call_args
        assert call_args.kwargs["index_config"].type == "lsh"
        assert call_args.kwargs["index_config"].lsh_num_tables == 16
        assert call_args.kwargs["index_config"].lsh_hyperplanes_per_table == 32
    
    def test_create_library_with_ivf_config(self):
        """Test library creation with IVF index configuration"""
        # Arrange
        ivf_library = Library(
            id=self.library_id,
            name="IVF Library",
            embedding_dim=512,
            index_config=IndexType(type="ivf"),
            metadata=LibraryMetadata(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        self.mock_library_svc.create.return_value = ivf_library
        
        request_data = {
            "name": "IVF Library",
            "embedding_dim": 512,
            "index_config": {
                "type": "ivf",
                "ivf": {
                    "num_centroids": 128,
                    "nprobe": 8,
                    "max_kmeans_iters": 10
                }
            },
            "metadata": {}
        }
        
        # Act
        response = self.client.post("/libraries", json=request_data)
        
        # Assert
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "IVF Library"
        assert data["embedding_dim"] == 512
        assert data["index_config"]["type"] == "ivf"
        
        self.mock_library_svc.create.assert_called_once()
        call_args = self.mock_library_svc.create.call_args
        assert call_args.kwargs["index_config"].type == "ivf"
        assert call_args.kwargs["index_config"].ivf_num_centroids == 128
        assert call_args.kwargs["index_config"].ivf_nprobe == 8
    
    def test_create_library_validation_error(self):
        """Test library creation with validation error from service"""
        # Arrange
        self.mock_library_svc.create.side_effect = ValidationError("Invalid embedding dimension")
        
        request_data = {
            "name": "Invalid Library",
            "embedding_dim": 128,  # Valid for FastAPI validation
            "index_config": {"type": "flat"},
            "metadata": {}
        }
        
        # Act
        response = self.client.post("/libraries", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error from service
        self.mock_library_svc.create.assert_called_once()
    
    def test_create_library_missing_required_fields(self):
        """Test library creation with missing required fields"""
        # Arrange
        request_data = {
            "name": "Test Library"
            # Missing embedding_dim
        }
        
        # Act
        response = self.client.post("/libraries", json=request_data)
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_library_svc.create.assert_not_called()
    
    def test_list_libraries_success(self):
        """Test successful library listing"""
        # Arrange
        library1 = Library(
            id=uuid4(),
            name="Library 1",
            embedding_dim=128,
            index_config=IndexType(type="flat"),
            metadata=LibraryMetadata(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        library2 = Library(
            id=uuid4(),
            name="Library 2",
            embedding_dim=256,
            index_config=IndexType(type="lsh"),
            metadata=LibraryMetadata(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        
        self.mock_library_svc.list.return_value = [library1, library2]
        
        # Act
        response = self.client.get("/libraries?limit=10&offset=0")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 2
        assert data["page"]["limit"] == 10
        assert data["page"]["offset"] == 0
        assert data["page"]["has_more"] == False
        
        assert data["items"][0]["name"] == "Library 1"
        assert data["items"][1]["name"] == "Library 2"
        
        self.mock_library_svc.list.assert_called_once()
    
    def test_list_libraries_with_pagination(self):
        """Test library listing with pagination"""
        # Arrange
        libraries = []
        for i in range(25):
            lib = Library(
                id=uuid4(),
                name=f"Library {i}",
                embedding_dim=128,
                index_config=IndexType(type="flat"),
                metadata=LibraryMetadata(),
                created_at=self.created_at,
                updated_at=self.updated_at,
                version=1
            )
            libraries.append(lib)
        
        self.mock_library_svc.list.return_value = libraries
        
        # Act
        response = self.client.get("/libraries?limit=10&offset=10")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) == 10
        assert data["page"]["limit"] == 10
        assert data["page"]["offset"] == 10
        assert data["page"]["has_more"] is True
        
        self.mock_library_svc.list.assert_called_once()
    
    def test_list_libraries_default_params(self):
        """Test library listing with default parameters"""
        # Arrange
        self.mock_library_svc.list.return_value = [self.mock_library]
        
        # Act
        response = self.client.get("/libraries")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["page"]["limit"] == 50  # Default limit
        assert data["page"]["offset"] == 0   # Default offset
        self.mock_library_svc.list.assert_called_once()
    
    def test_list_libraries_validation_error(self):
        """Test library listing with invalid parameters"""
        # Act
        response = self.client.get("/libraries?limit=2000&offset=-1")
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_library_svc.list.assert_not_called()
    
    def test_get_library_success(self):
        """Test successful library retrieval"""
        # Arrange
        self.mock_library_svc.get.return_value = self.mock_library
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(self.library_id)
        assert data["name"] == "Test Library"
        assert data["embedding_dim"] == 128
        assert data["version"] == 1
        
        self.mock_library_svc.get.assert_called_once_with(self.library_id)
    
    def test_get_library_not_found(self):
        """Test library retrieval when library doesn't exist"""
        # Arrange
        self.mock_library_svc.get.side_effect = NotFoundError(f"Library with id {self.library_id} not found")
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}")
        
        # Assert
        assert response.status_code == 404
        self.mock_library_svc.get.assert_called_once_with(self.library_id)
    
    def test_get_library_invalid_uuid(self):
        """Test library retrieval with invalid UUID"""
        # Act
        response = self.client.get("/libraries/invalid-uuid")
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_library_svc.get.assert_not_called()
    
    def test_update_index_config_success(self):
        """Test successful index configuration update"""
        # Arrange
        updated_library = Library(
            id=self.library_id,
            name="Test Library",
            embedding_dim=128,
            index_config=IndexType(type="lsh"),
            metadata=LibraryMetadata(description="Test library description"),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=2
        )
        self.mock_library_svc.update_config.return_value = updated_library
        
        request_data = {
            "type": "lsh",
            "lsh": {
                "num_tables": 16,
                "hyperplanes_per_table": 32
            }
        }
        
        # Act
        response = self.client.patch(f"/libraries/{self.library_id}/index-config", json=request_data)
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(self.library_id)
        assert data["index_config"]["type"] == "lsh"
        assert data["version"] == 2
        
        self.mock_library_svc.update_config.assert_called_once()
        call_args = self.mock_library_svc.update_config.call_args
        assert call_args[0][0] == self.library_id
        assert call_args[0][1].type == "lsh"
    
    def test_update_index_config_not_found(self):
        """Test index configuration update when library doesn't exist"""
        # Arrange
        self.mock_library_svc.update_config.side_effect = NotFoundError(f"Library with id {self.library_id} not found")
        
        request_data = {"type": "lsh"}
        
        # Act
        response = self.client.patch(f"/libraries/{self.library_id}/index-config", json=request_data)
        
        # Assert
        assert response.status_code == 404
        self.mock_library_svc.update_config.assert_called_once()
    
    def test_update_index_config_conflict(self):
        """Test index configuration update with conflict"""
        # Arrange
        self.mock_library_svc.update_config.side_effect = ConflictError("Library modified concurrently")
        
        request_data = {"type": "lsh"}
        
        # Act
        response = self.client.patch(f"/libraries/{self.library_id}/index-config", json=request_data)
        
        # Assert
        assert response.status_code == 409
        self.mock_library_svc.update_config.assert_called_once()
    
    def test_rebuild_index_success(self):
        """Test successful index rebuild"""
        # Arrange
        self.mock_library_svc.get.return_value = self.mock_library
        self.mock_library_svc.update_config.return_value = self.mock_library
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/rebuild-index")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["library_id"] == str(self.library_id)
        assert data["index_type"] == "flat"
        assert "rebuild_at" in data
        
        self.mock_library_svc.get.assert_called_once_with(self.library_id)
        self.mock_library_svc.update_config.assert_called_once_with(self.library_id, self.mock_library.index_config)
    
    def test_rebuild_index_not_found(self):
        """Test index rebuild when library doesn't exist"""
        # Arrange
        self.mock_library_svc.get.side_effect = NotFoundError(f"Library with id {self.library_id} not found")
        
        # Act
        response = self.client.post(f"/libraries/{self.library_id}/rebuild-index")
        
        # Assert
        assert response.status_code == 404
        self.mock_library_svc.get.assert_called_once_with(self.library_id)
        self.mock_library_svc.update_config.assert_not_called()
    
    def test_delete_library_success(self):
        """Test successful library deletion"""
        # Arrange
        self.mock_library_svc.delete.return_value = None
        
        # Act
        response = self.client.delete(f"/libraries/{self.library_id}")
        
        # Assert
        assert response.status_code == 204
        assert response.content == b""  # No content for 204
        self.mock_library_svc.delete.assert_called_once_with(self.library_id)
    
    def test_delete_library_not_found(self):
        """Test library deletion when library doesn't exist"""
        # Arrange
        self.mock_library_svc.delete.return_value = None  # Service returns silently
        
        # Act
        response = self.client.delete(f"/libraries/{self.library_id}")
        
        # Assert
        assert response.status_code == 204  # Still returns 204 even if not found
        self.mock_library_svc.delete.assert_called_once_with(self.library_id)
    
    def test_delete_library_invalid_uuid(self):
        """Test library deletion with invalid UUID"""
        # Act
        response = self.client.delete("/libraries/invalid-uuid")
        
        # Assert
        assert response.status_code == 422  # Validation error
        self.mock_library_svc.delete.assert_not_called()
    
    def test_library_response_structure(self):
        """Test that library response has correct structure"""
        # Arrange
        self.mock_library_svc.get.return_value = self.mock_library
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        # Check all required fields are present
        required_fields = [
            "id", "name", "embedding_dim", "index_config", 
            "metadata", "created_at", "updated_at", "version"
        ]
        for field in required_fields:
            assert field in data
        
        # Check field types
        assert isinstance(data["id"], str)  # UUID as string
        assert isinstance(data["name"], str)
        assert isinstance(data["embedding_dim"], int)
        assert isinstance(data["index_config"], dict)
        assert isinstance(data["metadata"], dict)
        assert isinstance(data["version"], int)
    
    def test_index_config_response_structure(self):
        """Test that index config response has correct structure"""
        # Arrange
        lsh_library = Library(
            id=self.library_id,
            name="LSH Library",
            embedding_dim=128,
            index_config=IndexType(type="lsh"),
            metadata=LibraryMetadata(),
            created_at=self.created_at,
            updated_at=self.updated_at,
            version=1
        )
        self.mock_library_svc.get.return_value = lsh_library
        
        # Act
        response = self.client.get(f"/libraries/{self.library_id}")
        
        # Assert
        assert response.status_code == 200
        data = response.json()
        
        index_config = data["index_config"]
        assert "type" in index_config
        assert index_config["type"] == "lsh"
