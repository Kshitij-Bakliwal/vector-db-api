"""
Unit tests for Chunk service
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.services.chunk import ChunkService
from vector_db_api.services.exceptions import NotFoundError, ValidationError, ConflictError
from vector_db_api.models.entities import Chunk, Document, Library
from vector_db_api.models.metadata import ChunkMetadata
from vector_db_api.models.indexing import IndexType


class TestChunkService:
    """Test cases for Chunk service"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_libs = Mock()
        self.mock_docs = Mock()
        self.mock_chunks = Mock()
        self.mock_locks = Mock()
        self.mock_indexes = Mock()
        
        self.service = ChunkService(
            self.mock_libs,
            self.mock_docs,
            self.mock_chunks,
            self.mock_locks,
            self.mock_indexes
        )
        
        self.library_id = uuid4()
        self.document_id = uuid4()
        self.chunk_id = uuid4()
        
        self.mock_library = Library(
            id=self.library_id,
            name="Test Library",
            embedding_dim=128,
            index_config=IndexType(type="flat"),
            metadata={}
        )
        
        self.mock_document = Document(
            id=self.document_id,
            library_id=self.library_id,
            metadata={},
            chunk_ids=[]
        )
        
        self.test_chunk_data = {
            "id": self.chunk_id,
            "library_id": self.library_id,
            "document_id": self.document_id,
            "text": "Test chunk text",
            "position": 0,
            "embedding": [0.1, 0.2, 0.3] * 42 + [0.1, 0.2],  # 128 dimensions
            "metadata": ChunkMetadata(page_number=1, token_count=10)
        }
    
    def test_upsert_chunk_new_success(self):
        """Test successful upsert of new chunk"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get_or_create.return_value = mock_index
        self.mock_chunks.get.return_value = None  # New chunk
        self.mock_chunks.add.return_value = None
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        result = self.service.upsert(mock_chunk)
        
        # Assert
        assert result == mock_chunk
        self.mock_libs.get.assert_called_once_with(self.library_id)
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_chunks.get.assert_called_once_with(self.chunk_id)
        self.mock_chunks.add.assert_called_once()
        self.mock_docs.update_on_version.assert_called_once()
        mock_index.add.assert_called_once_with(self.chunk_id, mock_chunk.embedding)
    
    def test_upsert_chunk_existing_success(self):
        """Test successful upsert of existing chunk"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        existing_chunk = Chunk(**self.test_chunk_data)
        existing_chunk.version = 1
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get_or_create.return_value = mock_index
        self.mock_chunks.get.return_value = existing_chunk  # Existing chunk
        self.mock_chunks.update_on_version.return_value = True
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        result = self.service.upsert(mock_chunk)
        
        # Assert
        assert result == mock_chunk
        self.mock_libs.get.assert_called_once_with(self.library_id)
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_chunks.get.assert_called_once_with(self.chunk_id)
        self.mock_chunks.update_on_version.assert_called_once()
        mock_index.update.assert_called_once_with(self.chunk_id, mock_chunk.embedding)
    
    def test_upsert_chunk_library_not_found(self):
        """Test upsert when library doesn't exist"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        self.mock_libs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Library with id {self.library_id} not found"):
            self.service.upsert(mock_chunk)
    
    def test_upsert_chunk_document_not_found(self):
        """Test upsert when document doesn't exist"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(ValidationError, match=f"Document with id {self.document_id} not found or not in library"):
            self.service.upsert(mock_chunk)
    
    def test_upsert_chunk_document_wrong_library(self):
        """Test upsert when document belongs to different library"""
        # Arrange
        wrong_library_id = uuid4()
        mock_chunk = Chunk(**self.test_chunk_data)
        wrong_doc = Document(id=self.document_id, library_id=wrong_library_id, metadata={}, chunk_ids=[])
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = wrong_doc
        
        # Act & Assert
        with pytest.raises(ValidationError, match=f"Document with id {self.document_id} not found or not in library"):
            self.service.upsert(mock_chunk)
    
    def test_upsert_chunk_embedding_dim_mismatch(self):
        """Test upsert when embedding dimension doesn't match library"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        mock_chunk.embedding = [0.1, 0.2, 0.3]  # Only 3 dimensions, should be 128
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = self.mock_document
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Embedding dim mismatch: got 3, expected 128"):
            self.service.upsert(mock_chunk)
    
    def test_upsert_chunk_concurrent_modification(self):
        """Test upsert with concurrent modification"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        existing_chunk = Chunk(**self.test_chunk_data)
        existing_chunk.version = 1
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get_or_create.return_value = mock_index
        self.mock_chunks.get.return_value = existing_chunk
        self.mock_chunks.update_on_version.return_value = False  # Concurrent modification
        
        # Act & Assert
        with pytest.raises(ConflictError, match=f"Chunk with id {self.chunk_id} modified concurrently during upsert"):
            self.service.upsert(mock_chunk)
    
    def test_upsert_chunk_document_concurrent_modification(self):
        """Test upsert with document concurrent modification"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        existing_chunk = Chunk(**self.test_chunk_data)
        existing_chunk.version = 1
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get_or_create.return_value = mock_index
        self.mock_chunks.get.return_value = existing_chunk
        self.mock_chunks.update_on_version.return_value = True
        self.mock_docs.update_on_version.return_value = False  # Document concurrent modification
        
        # Act & Assert
        with pytest.raises(ConflictError, match=f"Document with id {self.document_id} modified concurrently during upsert"):
            self.service.upsert(mock_chunk)
    
    def test_bulk_upsert_success(self):
        """Test successful bulk upsert of chunks"""
        # Arrange
        chunk1 = Chunk(
            id=uuid4(),
            library_id=self.library_id,
            document_id=self.document_id,
            text="First chunk",
            position=0,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2],  # 128 dimensions
            metadata=ChunkMetadata(page_number=1)
        )
        chunk2 = Chunk(
            id=uuid4(),
            library_id=self.library_id,
            document_id=self.document_id,
            text="Second chunk",
            position=1,
            embedding=[0.4, 0.5, 0.6] * 42 + [0.4, 0.5],  # 128 dimensions
            metadata=ChunkMetadata(page_number=2)
        )
        chunks = [chunk1, chunk2]
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get_or_create.return_value = mock_index
        self.mock_chunks.add.return_value = None
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        result = self.service.bulk_upsert(self.library_id, self.document_id, chunks)
        
        # Assert
        assert len(result) == 2
        assert result == chunks
        self.mock_libs.get.assert_called_once_with(self.library_id)
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        assert self.mock_chunks.add.call_count == 2
        assert mock_index.add.call_count == 2
        self.mock_docs.update_on_version.assert_called_once()
    
    def test_bulk_upsert_library_not_found(self):
        """Test bulk upsert when library doesn't exist"""
        # Arrange
        chunk = Chunk(**self.test_chunk_data)
        self.mock_libs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Library with id {self.library_id} not found"):
            self.service.bulk_upsert(self.library_id, self.document_id, [chunk])
    
    def test_bulk_upsert_document_not_found(self):
        """Test bulk upsert when document doesn't exist"""
        # Arrange
        chunk = Chunk(**self.test_chunk_data)
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(ValidationError, match=f"Document with id {self.document_id} not found or not in library"):
            self.service.bulk_upsert(self.library_id, self.document_id, [chunk])
    
    def test_bulk_upsert_embedding_dim_mismatch(self):
        """Test bulk upsert when embedding dimension doesn't match"""
        # Arrange
        chunk = Chunk(**self.test_chunk_data)
        chunk.embedding = [0.1, 0.2, 0.3]  # Only 3 dimensions, should be 128
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = self.mock_document
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Embedding dim mismatch: got 3, expected 128"):
            self.service.bulk_upsert(self.library_id, self.document_id, [chunk])
    
    def test_bulk_upsert_document_concurrent_modification(self):
        """Test bulk upsert with document concurrent modification"""
        # Arrange
        chunk = Chunk(**self.test_chunk_data)
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get_or_create.return_value = mock_index
        self.mock_chunks.add.return_value = None
        self.mock_docs.update_on_version.return_value = False  # Concurrent modification
        
        # Act & Assert
        with pytest.raises(ConflictError, match=f"Document with id {self.document_id} modified concurrently during bulk upsert"):
            self.service.bulk_upsert(self.library_id, self.document_id, [chunk])
    
    def test_delete_chunk_success(self):
        """Test successful chunk deletion"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_chunks.get.return_value = mock_chunk
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        self.mock_chunks.delete.return_value = True
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        self.service.delete(self.library_id, self.chunk_id)
        
        # Assert
        self.mock_chunks.get.assert_called_once_with(self.chunk_id)
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_indexes.get.assert_called_once_with(self.library_id)
        mock_index.remove.assert_called_once_with(self.chunk_id)
        self.mock_chunks.delete.assert_called_once_with(self.chunk_id)
        self.mock_docs.update_on_version.assert_called_once()
    
    def test_delete_chunk_not_found(self):
        """Test chunk deletion when chunk doesn't exist"""
        # Arrange
        self.mock_chunks.get.return_value = None
        
        # Act
        self.service.delete(self.library_id, self.chunk_id)
        
        # Assert - should return silently when chunk doesn't exist
        self.mock_chunks.get.assert_called_once_with(self.chunk_id)
        # Should not call any other methods
        self.mock_docs.get.assert_not_called()
        self.mock_locks.lock_for_library.assert_not_called()
        self.mock_indexes.get.assert_not_called()
        self.mock_chunks.delete.assert_not_called()
        self.mock_docs.update_on_version.assert_not_called()
    
    def test_delete_chunk_wrong_library(self):
        """Test chunk deletion when chunk belongs to different library"""
        # Arrange
        wrong_library_id = uuid4()
        mock_chunk = Chunk(**self.test_chunk_data)
        mock_chunk.library_id = wrong_library_id
        
        self.mock_chunks.get.return_value = mock_chunk
        
        # Act
        self.service.delete(self.library_id, self.chunk_id)
        
        # Assert - should return silently when chunk belongs to different library
        self.mock_chunks.get.assert_called_once_with(self.chunk_id)
        # Should not call any other methods
        self.mock_docs.get.assert_not_called()
        self.mock_locks.lock_for_library.assert_not_called()
        self.mock_indexes.get.assert_not_called()
        self.mock_chunks.delete.assert_not_called()
        self.mock_docs.update_on_version.assert_not_called()
    
    def test_delete_chunk_no_embedding(self):
        """Test chunk deletion when chunk has no embedding"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        mock_chunk.embedding = None  # No embedding
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_chunks.get.return_value = mock_chunk
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        self.mock_chunks.delete.return_value = True
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        self.service.delete(self.library_id, self.chunk_id)
        
        # Assert
        self.mock_chunks.get.assert_called_once_with(self.chunk_id)
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_indexes.get.assert_called_once_with(self.library_id)
        mock_index.remove.assert_not_called()  # Should not remove from index
        self.mock_chunks.delete.assert_called_once_with(self.chunk_id)
        self.mock_docs.update_on_version.assert_called_once()
    
    def test_delete_chunk_no_index(self):
        """Test chunk deletion when no index exists"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        mock_lock = Mock()
        
        self.mock_chunks.get.return_value = mock_chunk
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = None  # No index
        self.mock_chunks.delete.return_value = True
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        self.service.delete(self.library_id, self.chunk_id)
        
        # Assert
        self.mock_chunks.get.assert_called_once_with(self.chunk_id)
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_indexes.get.assert_called_once_with(self.library_id)
        self.mock_chunks.delete.assert_called_once_with(self.chunk_id)
        self.mock_docs.update_on_version.assert_called_once()
    
    def test_delete_chunk_document_concurrent_modification(self):
        """Test chunk deletion with document concurrent modification"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_chunks.get.return_value = mock_chunk
        self.mock_docs.get.return_value = self.mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        self.mock_chunks.delete.return_value = True
        self.mock_docs.update_on_version.return_value = False  # Concurrent modification
        
        # Act & Assert
        with pytest.raises(ConflictError, match=f"Document with id {self.document_id} modified concurrently during delete"):
            self.service.delete(self.library_id, self.chunk_id)
    
    def test_delete_chunk_not_in_document_chunk_ids(self):
        """Test chunk deletion when chunk is not in document's chunk_ids"""
        # Arrange
        mock_chunk = Chunk(**self.test_chunk_data)
        mock_document = Document(
            id=self.document_id,
            library_id=self.library_id,
            metadata={},
            chunk_ids=[]  # Empty chunk_ids
        )
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_chunks.get.return_value = mock_chunk
        self.mock_docs.get.return_value = mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        self.mock_chunks.delete.return_value = True
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        self.service.delete(self.library_id, self.chunk_id)
        
        # Assert
        self.mock_chunks.get.assert_called_once_with(self.chunk_id)
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_indexes.get.assert_called_once_with(self.library_id)
        mock_index.remove.assert_called_once_with(self.chunk_id)
        self.mock_chunks.delete.assert_called_once_with(self.chunk_id)
        self.mock_docs.update_on_version.assert_called_once()
        # chunk_ids should remain empty since chunk_id was not in the list
        assert len(mock_document.chunk_ids) == 0