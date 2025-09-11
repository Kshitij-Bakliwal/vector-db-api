"""
Unit tests for Library service
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.services.library import LibraryService
from vector_db_api.services.exceptions import NotFoundError, ValidationError, ConflictError
from vector_db_api.models.entities import Library
from vector_db_api.models.metadata import LibraryMetadata
from vector_db_api.models.indexing import IndexType


class TestLibraryService:
    """Test cases for Library service"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_libs = Mock()
        self.mock_docs = Mock()
        self.mock_chunks = Mock()
        self.mock_locks = Mock()
        self.mock_indexes = Mock()
        
        self.service = LibraryService(
            self.mock_libs,
            self.mock_docs,
            self.mock_chunks,
            self.mock_locks,
            self.mock_indexes
        )
        
        self.test_library_data = {
            "name": "Test Library",
            "embedding_dim": 128,
            "index_config": IndexType(type="flat"),
            "metadata": LibraryMetadata(description="Test library")
        }
    
    def test_create_library_success(self):
        """Test successful library creation"""
        # Arrange
        mock_library = Library(**self.test_library_data)
        self.mock_libs.add.return_value = None
        self.mock_libs.get.return_value = mock_library
        
        # Act
        result = self.service.create(**self.test_library_data)
        
        # Assert
        assert result.name == "Test Library"
        assert result.embedding_dim == 128
        assert result.index_config.type == "flat"
        self.mock_libs.add.assert_called_once()
        self.mock_indexes.get_or_create.assert_called_once_with(result)
    
    def test_create_library_with_invalid_embedding_dim(self):
        """Test library creation with invalid embedding dimension"""
        # Arrange
        invalid_data = self.test_library_data.copy()
        invalid_data["embedding_dim"] = 0
        
        # Act & Assert
        with pytest.raises(Exception):  # Pydantic validation error
            self.service.create(**invalid_data)
    
    def test_create_library_with_negative_embedding_dim(self):
        """Test library creation with negative embedding dimension"""
        # Arrange
        invalid_data = self.test_library_data.copy()
        invalid_data["embedding_dim"] = -1
        
        # Act & Assert
        with pytest.raises(Exception):  # Pydantic validation error
            self.service.create(**invalid_data)
    
    def test_get_library_success(self):
        """Test successful library retrieval"""
        # Arrange
        mock_library = Library(**self.test_library_data)
        self.mock_libs.get.return_value = mock_library
        
        # Act
        result = self.service.get(mock_library.id)
        
        # Assert
        assert result == mock_library
        self.mock_libs.get.assert_called_once_with(mock_library.id)
    
    def test_get_library_not_found(self):
        """Test library retrieval when library doesn't exist"""
        # Arrange
        library_id = uuid4()
        self.mock_libs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Library with id {library_id} not found"):
            self.service.get(library_id)
    
    def test_list_libraries_success(self):
        """Test successful library listing"""
        # Arrange
        mock_libraries = [
            Library(**self.test_library_data),
            Library(name="Library 2", embedding_dim=256, metadata=LibraryMetadata())
        ]
        self.mock_libs.list.return_value = mock_libraries
        
        # Act
        result = self.service.list()
        
        # Assert
        assert len(result) == 2
        assert result == mock_libraries
        self.mock_libs.list.assert_called_once()
    
    def test_update_config_success(self):
        """Test successful library config update"""
        # Arrange
        library_id = uuid4()
        mock_library = Library(id=library_id, **self.test_library_data)
        new_config = IndexType(type="lsh")
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_libs.get.return_value = mock_library
        self.mock_libs.update_on_version.return_value = True
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.create_index.return_value = mock_index
        self.mock_chunks.list_by_library.return_value = []
        
        # Act
        result = self.service.update_config(library_id, new_config)
        
        # Assert
        assert result == mock_library
        self.mock_libs.get.assert_called_once_with(library_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_libs.update_on_version.assert_called_once()
    
    def test_update_config_not_found(self):
        """Test library config update when library doesn't exist"""
        # Arrange
        library_id = uuid4()
        new_config = IndexType(type="lsh")
        self.mock_libs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Library with id {library_id} not found"):
            self.service.update_config(library_id, new_config)
    
    def test_delete_library_success(self):
        """Test successful library deletion"""
        # Arrange
        library_id = uuid4()
        mock_library = Library(id=library_id, **self.test_library_data)
        mock_lock = Mock()
        
        self.mock_libs.get.return_value = mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_docs.list_by_library.return_value = []  # No documents
        self.mock_libs.delete.return_value = True
        
        # Act
        self.service.delete(library_id)
        
        # Assert
        self.mock_libs.get.assert_called_once_with(library_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_indexes.remove.assert_called_once_with(library_id)
        self.mock_libs.delete.assert_called_once_with(library_id)
    
    def test_delete_library_not_found(self):
        """Test library deletion when library doesn't exist"""
        # Arrange
        library_id = uuid4()
        self.mock_libs.get.return_value = None
        
        # Act
        self.service.delete(library_id)
        
        # Assert - should return silently when library doesn't exist
        self.mock_libs.get.assert_called_once_with(library_id)
        # Should not call any other methods
        self.mock_locks.lock_for_library.assert_not_called()
        self.mock_indexes.remove.assert_not_called()
        self.mock_libs.delete.assert_not_called()
    
    def test_delete_library_with_documents(self):
        """Test library deletion when library has documents"""
        # Arrange
        library_id = uuid4()
        mock_library = Library(id=library_id, **self.test_library_data)
        mock_lock = Mock()
        mock_doc = Mock()
        mock_doc.id = uuid4()
        
        self.mock_libs.get.return_value = mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        # Create a mock that returns documents on first call, empty list on subsequent calls
        class MockListByLibrary:
            def __init__(self):
                self.call_count = 0
            
            def __call__(self, lib_id, limit=1000, offset=0):
                self.call_count += 1
                if self.call_count == 1:
                    return [mock_doc]
                else:
                    return []
        
        self.mock_docs.list_by_library = MockListByLibrary()
        self.mock_chunks.delete_by_document.return_value = 0
        self.mock_docs.delete.return_value = True
        self.mock_libs.delete.return_value = True
        
        # Act
        self.service.delete(library_id)
        
        # Assert - should delete documents and chunks
        self.mock_libs.get.assert_called_once_with(library_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        # Should be called twice: once with offset=0, once with offset=1000
        assert self.mock_docs.list_by_library.call_count == 2
        self.mock_chunks.delete_by_document.assert_called_once_with(mock_doc.id)
        self.mock_docs.delete.assert_called_once_with(mock_doc.id)
        self.mock_indexes.remove.assert_called_once_with(library_id)
        self.mock_libs.delete.assert_called_once_with(library_id)
    
    def test_update_config_with_chunks(self):
        """Test config update with existing chunks"""
        # Arrange
        library_id = uuid4()
        mock_library = Library(id=library_id, **self.test_library_data)
        new_config = IndexType(type="lsh")
        mock_lock = Mock()
        mock_index = Mock()
        mock_chunk1 = Mock()
        mock_chunk1.id = uuid4()
        mock_chunk1.embedding = [0.1, 0.2, 0.3]
        mock_chunk2 = Mock()
        mock_chunk2.id = uuid4()
        mock_chunk2.embedding = [0.4, 0.5, 0.6]
        
        self.mock_libs.get.return_value = mock_library
        self.mock_libs.update_on_version.return_value = True
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.create_index.return_value = mock_index
        # Mock to return chunks on first call, empty list on subsequent calls
        self.mock_chunks.list_by_library.side_effect = [[mock_chunk1, mock_chunk2], []]
        
        # Act
        result = self.service.update_config(library_id, new_config)
        
        # Assert
        assert result == mock_library
        self.mock_libs.get.assert_called_once_with(library_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_libs.update_on_version.assert_called_once()
        self.mock_indexes.create_index.assert_called_once_with(new_config, mock_library.embedding_dim)
        # Verify rebuild is called with the chunks
        mock_index.rebuild.assert_called_once_with([(mock_chunk1.id, mock_chunk1.embedding), (mock_chunk2.id, mock_chunk2.embedding)])
        self.mock_indexes.swap.assert_called_once_with(library_id, mock_index)
