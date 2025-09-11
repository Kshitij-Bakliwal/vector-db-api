"""
Unit tests for Document service
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.services.document import DocumentService
from vector_db_api.services.exceptions import NotFoundError, ValidationError, ConflictError
from vector_db_api.models.entities import Document, Chunk, Library
from vector_db_api.models.metadata import DocumentMetadata, ChunkMetadata
from vector_db_api.models.indexing import IndexType


class TestDocumentService:
    """Test cases for Document service"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_libs = Mock()
        self.mock_docs = Mock()
        self.mock_chunks = Mock()
        self.mock_locks = Mock()
        self.mock_indexes = Mock()
        
        self.service = DocumentService(
            self.mock_libs,
            self.mock_docs,
            self.mock_chunks,
            self.mock_locks,
            self.mock_indexes
        )
        
        self.library_id = uuid4()
        self.document_id = uuid4()
        self.mock_library = Library(
            id=self.library_id,
            name="Test Library",
            embedding_dim=128,
            index_config=IndexType(type="flat"),
            metadata={}
        )
        
        self.test_document_data = {
            "library_id": self.library_id,
            "metadata": DocumentMetadata(title="Test Document", summary="Test document summary")
        }
    
    def test_create_document_success(self):
        """Test successful document creation"""
        # Arrange
        mock_document = Document(**self.test_document_data)
        mock_lock = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_docs.add.return_value = None
        
        # Act
        result = self.service.create(self.library_id, self.test_document_data["metadata"])
        
        # Assert
        assert result.library_id == self.library_id
        assert result.metadata.title == "Test Document"
        assert result.metadata.summary == "Test document summary"
        self.mock_libs.get.assert_called_once_with(self.library_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_docs.add.assert_called_once()
    
    def test_create_document_with_dict_metadata(self):
        """Test document creation with dictionary metadata"""
        # Arrange
        metadata_dict = {"title": "Test Document", "summary": "Test document summary", "author": "Test Author"}
        mock_document = Document(library_id=self.library_id, metadata=DocumentMetadata(**metadata_dict))
        mock_lock = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_docs.add.return_value = None
        
        # Act
        result = self.service.create(self.library_id, metadata_dict)
        
        # Assert
        assert result.library_id == self.library_id
        assert result.metadata.title == "Test Document"
        assert result.metadata.summary == "Test document summary"
        assert result.metadata.author == "Test Author"
        self.mock_libs.get.assert_called_once_with(self.library_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_docs.add.assert_called_once()
    
    def test_create_document_with_none_metadata(self):
        """Test document creation with None metadata"""
        # Arrange
        mock_document = Document(library_id=self.library_id, metadata=DocumentMetadata())
        mock_lock = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_docs.add.return_value = None
        
        # Act
        result = self.service.create(self.library_id, None)
        
        # Assert
        assert result.library_id == self.library_id
        assert result.metadata == DocumentMetadata()
        self.mock_libs.get.assert_called_once_with(self.library_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_docs.add.assert_called_once()
    
    def test_create_document_library_not_found(self):
        """Test document creation when library doesn't exist"""
        # Arrange
        self.mock_libs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Library with id {self.library_id} not found"):
            self.service.create(self.library_id, self.test_document_data["metadata"])
    
    def test_create_with_chunks_success(self):
        """Test successful document creation with chunks"""
        # Arrange
        mock_document = Document(**self.test_document_data)
        mock_lock = Mock()
        mock_index = Mock()
        
        chunk1 = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="First chunk",
            position=0,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2]  # 128 dimensions
        )
        chunk2 = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="Second chunk",
            position=1,
            embedding=[0.4, 0.5, 0.6] * 42 + [0.4, 0.5]  # 128 dimensions
        )
        chunks = [chunk1, chunk2]
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get_or_create.return_value = mock_index
        self.mock_docs.add.return_value = None
        self.mock_docs.update_on_version.return_value = True
        self.mock_chunks.add.return_value = None
        
        # Act
        result = self.service.create_with_chunks(self.library_id, chunks, self.test_document_data["metadata"])
        
        # Assert
        assert result.library_id == self.library_id
        assert result.metadata.title == "Test Document"
        assert result.metadata.summary == "Test document summary"
        self.mock_libs.get.assert_called_once_with(self.library_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_docs.add.assert_called_once()
        self.mock_docs.update_on_version.assert_called_once()
        assert self.mock_chunks.add.call_count == 2
        assert mock_index.add.call_count == 2
    
    def test_create_with_chunks_embedding_dim_mismatch(self):
        """Test document creation with chunks having wrong embedding dimension"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        
        chunk = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="Wrong dimension chunk",
            position=0,
            embedding=[0.1, 0.2, 0.3]  # Only 3 dimensions, should be 128
        )
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get_or_create.return_value = mock_index
        self.mock_docs.add.return_value = None
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Embedding dim mismatch: got 3, expected 128"):
            self.service.create_with_chunks(self.library_id, [chunk], self.test_document_data["metadata"])
    
    def test_create_with_chunks_library_not_found(self):
        """Test document creation with chunks when library doesn't exist"""
        # Arrange
        chunk = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="Test chunk",
            position=0,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2]  # 128 dimensions
        )
        
        self.mock_libs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Library with id {self.library_id} not found"):
            self.service.create_with_chunks(self.library_id, [chunk], self.test_document_data["metadata"])
    
    def test_get_document_success(self):
        """Test successful document retrieval"""
        # Arrange
        mock_document = Document(id=self.document_id, **self.test_document_data)
        self.mock_docs.get.return_value = mock_document
        
        # Act
        result = self.service.get(self.library_id, self.document_id)
        
        # Assert
        assert result == mock_document
        self.mock_docs.get.assert_called_once_with(self.document_id)
    
    def test_get_document_not_found(self):
        """Test document retrieval when document doesn't exist"""
        # Arrange
        self.mock_docs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Document with id {self.document_id} not found"):
            self.service.get(self.library_id, self.document_id)
    
    def test_get_document_wrong_library(self):
        """Test document retrieval when document belongs to different library"""
        # Arrange
        wrong_library_id = uuid4()
        mock_document = Document(id=self.document_id, library_id=wrong_library_id, metadata=DocumentMetadata())
        self.mock_docs.get.return_value = mock_document
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Document with id {self.document_id} not found"):
            self.service.get(self.library_id, self.document_id)
    
    def test_list_documents_success(self):
        """Test successful document listing"""
        # Arrange
        mock_documents = [
            Document(id=uuid4(), **self.test_document_data),
            Document(id=uuid4(), library_id=self.library_id, metadata=DocumentMetadata())
        ]
        self.mock_docs.list_by_library.return_value = mock_documents
        
        # Act
        result = self.service.list(self.library_id, limit=50, offset=10)
        
        # Assert
        assert len(result) == 2
        assert result == mock_documents
        self.mock_docs.list_by_library.assert_called_once_with(self.library_id, 50, 10, None, None, "updated_at", "desc")
    
    def test_update_metadata_success(self):
        """Test successful metadata update"""
        # Arrange
        mock_document = Document(id=self.document_id, **self.test_document_data)
        new_metadata = DocumentMetadata(title="Updated Title", summary="Updated summary", author="New Author")
        mock_lock = Mock()
        
        self.mock_docs.get.return_value = mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        result = self.service.update_metadata(self.library_id, self.document_id, new_metadata)
        
        # Assert
        assert result == mock_document
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_docs.update_on_version.assert_called_once()
    
    def test_update_metadata_document_not_found(self):
        """Test metadata update when document doesn't exist"""
        # Arrange
        new_metadata = DocumentMetadata(title="Updated Title", summary="Updated summary")
        self.mock_docs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Document with id {self.document_id} not found"):
            self.service.update_metadata(self.library_id, self.document_id, new_metadata)
    
    def test_update_metadata_concurrent_modification(self):
        """Test metadata update with concurrent modification"""
        # Arrange
        mock_document = Document(id=self.document_id, **self.test_document_data)
        new_metadata = DocumentMetadata(title="Updated Title", summary="Updated summary")
        mock_lock = Mock()
        
        self.mock_docs.get.return_value = mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_docs.update_on_version.return_value = False  # Concurrent modification
        
        # Act & Assert
        with pytest.raises(ConflictError, match=f"Document with id {self.document_id} modified concurrently during update"):
            self.service.update_metadata(self.library_id, self.document_id, new_metadata)
    
    def test_delete_document_success(self):
        """Test successful document deletion"""
        # Arrange
        mock_document = Document(id=self.document_id, **self.test_document_data)
        mock_lock = Mock()
        mock_index = Mock()
        mock_chunk = Mock()
        mock_chunk.id = uuid4()
        mock_chunk.embedding = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]  # 128 dimensions
        
        self.mock_docs.get.return_value = mock_document
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        self.mock_chunks.list_by_document.return_value = [mock_chunk]
        self.mock_chunks.delete_by_document.return_value = 1
        self.mock_docs.delete.return_value = True
        
        # Act
        self.service.delete(self.library_id, self.document_id)
        
        # Assert
        self.mock_docs.get.assert_called_once_with(self.document_id)
        mock_lock.acquire_write.assert_called_once()
        mock_lock.release_write.assert_called_once()
        self.mock_indexes.get.assert_called_once_with(self.library_id)
        self.mock_chunks.list_by_document.assert_called_once_with(self.document_id, limit=10**9, offset=0)
        mock_index.remove.assert_called_once_with(mock_chunk.id)
        self.mock_chunks.delete_by_document.assert_called_once_with(self.document_id)
        self.mock_docs.delete.assert_called_once_with(self.document_id)
    
    def test_delete_document_not_found(self):
        """Test document deletion when document doesn't exist"""
        # Arrange
        self.mock_docs.get.return_value = None
        
        # Act
        self.service.delete(self.library_id, self.document_id)
        
        # Assert - should return silently when document doesn't exist
        self.mock_docs.get.assert_called_once_with(self.document_id)
        # Should not call any other methods
        self.mock_locks.lock_for_library.assert_not_called()
        self.mock_indexes.get.assert_not_called()
        self.mock_chunks.delete_by_document.assert_not_called()
        self.mock_docs.delete.assert_not_called()
    
    def test_delete_document_wrong_library(self):
        """Test document deletion when document belongs to different library"""
        # Arrange
        wrong_library_id = uuid4()
        mock_document = Document(id=self.document_id, library_id=wrong_library_id, metadata=DocumentMetadata())
        self.mock_docs.get.return_value = mock_document
        
        # Act
        self.service.delete(self.library_id, self.document_id)
        
        # Assert - should return silently when document belongs to different library
        self.mock_docs.get.assert_called_once_with(self.document_id)
        # Should not call any other methods
        self.mock_locks.lock_for_library.assert_not_called()
        self.mock_indexes.get.assert_not_called()
        self.mock_chunks.delete_by_document.assert_not_called()
        self.mock_docs.delete.assert_not_called()
    
    def test_move_to_library_success(self):
        """Test successful document move to different library"""
        # Arrange
        src_lib_id = uuid4()
        dst_lib_id = uuid4()
        mock_src_lib = Library(id=src_lib_id, name="Source", embedding_dim=128, index_config=IndexType(), metadata={})
        mock_dst_lib = Library(id=dst_lib_id, name="Destination", embedding_dim=128, index_config=IndexType(), metadata={})
        mock_document = Document(id=self.document_id, library_id=src_lib_id, metadata=DocumentMetadata())
        mock_src_lock = Mock()
        mock_dst_lock = Mock()
        mock_src_index = Mock()
        mock_dst_index = Mock()
        mock_chunk = Mock()
        mock_chunk.id = uuid4()
        mock_chunk.embedding = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]  # 128 dimensions
        mock_chunk.version = 1
        
        self.mock_libs.get.side_effect = [mock_src_lib, mock_dst_lib]
        self.mock_locks.lock_for_library.side_effect = [mock_src_lock, mock_dst_lock]
        self.mock_docs.get.return_value = mock_document
        self.mock_indexes.get.return_value = mock_src_index
        self.mock_indexes.get_or_create.return_value = mock_dst_index
        self.mock_chunks.list_by_document.return_value = [mock_chunk]
        self.mock_chunks.update_on_version.return_value = True
        self.mock_docs.update_on_version.return_value = True
        
        # Act
        result = self.service.move_to_library(self.document_id, src_lib_id, dst_lib_id)
        
        # Assert
        assert result == mock_document
        assert result.library_id == dst_lib_id
        self.mock_libs.get.assert_any_call(src_lib_id)
        self.mock_libs.get.assert_any_call(dst_lib_id)
        mock_src_lock.acquire_write.assert_called_once()
        mock_dst_lock.acquire_write.assert_called_once()
        mock_src_lock.release_write.assert_called_once()
        mock_dst_lock.release_write.assert_called_once()
        self.mock_docs.get.assert_called_once_with(self.document_id)
        self.mock_indexes.get.assert_called_once_with(src_lib_id)
        self.mock_indexes.get_or_create.assert_called_once_with(mock_dst_lib)
        self.mock_chunks.list_by_document.assert_called_once_with(self.document_id, limit=10**9, offset=0)
        mock_src_index.remove.assert_called_once_with(mock_chunk.id)
        self.mock_chunks.update_on_version.assert_called_once()
        mock_dst_index.add.assert_called_once_with(mock_chunk.id, mock_chunk.embedding)
        self.mock_docs.update_on_version.assert_called_once()
    
    def test_move_to_library_same_library(self):
        """Test document move when source and destination are the same"""
        # Arrange
        lib_id = uuid4()
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Source and destination libraries are the same"):
            self.service.move_to_library(self.document_id, lib_id, lib_id)
    
    def test_move_to_library_source_not_found(self):
        """Test document move when source library doesn't exist"""
        # Arrange
        src_lib_id = uuid4()
        dst_lib_id = uuid4()
        
        self.mock_libs.get.side_effect = [None, Mock()]  # Source library not found
        
        # Act & Assert
        with pytest.raises(NotFoundError, match="Source or destination library not found"):
            self.service.move_to_library(self.document_id, src_lib_id, dst_lib_id)
    
    def test_move_to_library_destination_not_found(self):
        """Test document move when destination library doesn't exist"""
        # Arrange
        src_lib_id = uuid4()
        dst_lib_id = uuid4()
        
        self.mock_libs.get.side_effect = [Mock(), None]  # Destination library not found
        
        # Act & Assert
        with pytest.raises(NotFoundError, match="Source or destination library not found"):
            self.service.move_to_library(self.document_id, src_lib_id, dst_lib_id)
    
    def test_move_to_library_document_not_found(self):
        """Test document move when document doesn't exist in source library"""
        # Arrange
        src_lib_id = uuid4()
        dst_lib_id = uuid4()
        mock_src_lib = Library(id=src_lib_id, name="Source", embedding_dim=128, index_config=IndexType(), metadata={})
        mock_dst_lib = Library(id=dst_lib_id, name="Destination", embedding_dim=128, index_config=IndexType(), metadata={})
        mock_src_lock = Mock()
        mock_dst_lock = Mock()
        
        self.mock_libs.get.side_effect = [mock_src_lib, mock_dst_lib]
        self.mock_locks.lock_for_library.side_effect = [mock_src_lock, mock_dst_lock]
        self.mock_docs.get.return_value = None  # Document not found
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Document with id {self.document_id} not found in source library"):
            self.service.move_to_library(self.document_id, src_lib_id, dst_lib_id)
    
    def test_move_to_library_embedding_dim_mismatch(self):
        """Test document move when embedding dimensions don't match"""
        # Arrange
        src_lib_id = uuid4()
        dst_lib_id = uuid4()
        mock_src_lib = Library(id=src_lib_id, name="Source", embedding_dim=128, index_config=IndexType(), metadata={})
        mock_dst_lib = Library(id=dst_lib_id, name="Destination", embedding_dim=256, index_config=IndexType(), metadata={})  # Different embedding dim
        mock_document = Document(id=self.document_id, library_id=src_lib_id, metadata=DocumentMetadata())
        mock_src_lock = Mock()
        mock_dst_lock = Mock()
        mock_src_index = Mock()
        mock_dst_index = Mock()
        mock_chunk = Mock()
        mock_chunk.id = uuid4()
        mock_chunk.embedding = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]  # 128 dimensions
        mock_chunk.version = 1
        
        self.mock_libs.get.side_effect = [mock_src_lib, mock_dst_lib]
        self.mock_locks.lock_for_library.side_effect = [mock_src_lock, mock_dst_lock]
        self.mock_docs.get.return_value = mock_document
        self.mock_indexes.get.return_value = mock_src_index
        self.mock_indexes.get_or_create.return_value = mock_dst_index
        self.mock_chunks.list_by_document.return_value = [mock_chunk]
        self.mock_chunks.update_on_version.return_value = True
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Embedding dim mismatch for destination library"):
            self.service.move_to_library(self.document_id, src_lib_id, dst_lib_id)