"""
Unit tests for Search service
"""

import pytest
from uuid import uuid4
from unittest.mock import Mock, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.services.search import SearchService
from vector_db_api.services.exceptions import NotFoundError, ValidationError
from vector_db_api.models.entities import Library, Chunk
from vector_db_api.models.metadata import ChunkMetadata
from vector_db_api.models.indexing import IndexType


class TestSearchService:
    """Test cases for Search service"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.mock_libs = Mock()
        self.mock_chunks = Mock()
        self.mock_locks = Mock()
        self.mock_indexes = Mock()
        
        self.service = SearchService(
            self.mock_libs,
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
        
        self.query_embedding = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]  # 128 dimensions
        
        self.mock_chunk = Chunk(
            id=self.chunk_id,
            library_id=self.library_id,
            document_id=self.document_id,
            text="Test chunk text",
            position=0,
            embedding=[0.4, 0.5, 0.6] * 42 + [0.4, 0.5],  # 128 dimensions
            metadata=ChunkMetadata(page_number=1, token_count=10)
        )
    
    def test_query_success(self):
        """Test successful search query"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        search_results = [(self.chunk_id, 0.95), (uuid4(), 0.87)]
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        mock_index.search.return_value = search_results
        self.mock_chunks.get.return_value = self.mock_chunk
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding, k=5)
        
        # Assert
        assert len(result) == 2
        assert result[0]["chunk_id"] == self.chunk_id
        assert result[0]["document_id"] == self.document_id
        assert result[0]["score"] == 0.95
        assert result[0]["text"] == "Test chunk text"
        assert result[0]["metadata"]["page_number"] == 1
        assert result[0]["metadata"]["token_count"] == 10
        
        self.mock_libs.get.assert_called_once_with(self.library_id)
        mock_lock.acquire_read.assert_called_once()
        mock_lock.release_read.assert_called_once()
        self.mock_indexes.get.assert_called_once_with(self.library_id)
        mock_index.search.assert_called_once_with(self.query_embedding, 5, "cosine")
        assert self.mock_chunks.get.call_count == 2
    
    def test_query_with_default_k(self):
        """Test search query with default k value"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        search_results = [(self.chunk_id, 0.95)]
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        mock_index.search.return_value = search_results
        self.mock_chunks.get.return_value = self.mock_chunk
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding)
        
        # Assert
        assert len(result) == 1
        mock_index.search.assert_called_once_with(self.query_embedding, 10, "cosine")  # Default k=10
    
    def test_query_library_not_found(self):
        """Test search query when library doesn't exist"""
        # Arrange
        self.mock_libs.get.return_value = None
        
        # Act & Assert
        with pytest.raises(NotFoundError, match=f"Library with id {self.library_id} not found"):
            self.service.query(self.library_id, self.query_embedding)
    
    def test_query_embedding_dim_mismatch(self):
        """Test search query when embedding dimension doesn't match library"""
        # Arrange
        wrong_embedding = [0.1, 0.2, 0.3]  # Only 3 dimensions, should be 128
        self.mock_libs.get.return_value = self.mock_library
        
        # Act & Assert
        with pytest.raises(ValidationError, match="Embedding dim mismatch: got 3, expected 128"):
            self.service.query(self.library_id, wrong_embedding)
    
    def test_query_no_index(self):
        """Test search query when no index exists"""
        # Arrange
        mock_lock = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = None  # No index
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding)
        
        # Assert
        assert result == []
        self.mock_libs.get.assert_called_once_with(self.library_id)
        mock_lock.acquire_read.assert_called_once()
        mock_lock.release_read.assert_called_once()
        self.mock_indexes.get.assert_called_once_with(self.library_id)
        self.mock_chunks.get.assert_not_called()
    
    def test_query_empty_search_results(self):
        """Test search query when index returns empty results"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        mock_index.search.return_value = []  # Empty results
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding)
        
        # Assert
        assert result == []
        self.mock_libs.get.assert_called_once_with(self.library_id)
        mock_lock.acquire_read.assert_called_once()
        mock_lock.release_read.assert_called_once()
        self.mock_indexes.get.assert_called_once_with(self.library_id)
        mock_index.search.assert_called_once_with(self.query_embedding, 10, "cosine")
        self.mock_chunks.get.assert_not_called()
    
    def test_query_chunk_not_found(self):
        """Test search query when chunk from search results doesn't exist"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        non_existent_chunk_id = uuid4()
        search_results = [(self.chunk_id, 0.95), (non_existent_chunk_id, 0.87)]
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        mock_index.search.return_value = search_results
        # First chunk exists, second doesn't
        self.mock_chunks.get.side_effect = [self.mock_chunk, None]
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding)
        
        # Assert
        assert len(result) == 1  # Only the existing chunk should be returned
        assert result[0]["chunk_id"] == self.chunk_id
        assert result[0]["score"] == 0.95
        assert self.mock_chunks.get.call_count == 2
    
    def test_query_multiple_chunks(self):
        """Test search query with multiple chunks in results"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        chunk_id_2 = uuid4()
        chunk_id_3 = uuid4()
        
        chunk_2 = Chunk(
            id=chunk_id_2,
            library_id=self.library_id,
            document_id=self.document_id,
            text="Second chunk text",
            position=1,
            embedding=[0.7, 0.8, 0.9] * 42 + [0.7, 0.8],
            metadata=ChunkMetadata(page_number=2, token_count=15)
        )
        
        chunk_3 = Chunk(
            id=chunk_id_3,
            library_id=self.library_id,
            document_id=self.document_id,
            text="Third chunk text",
            position=2,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            metadata=ChunkMetadata(page_number=3, token_count=20)
        )
        
        search_results = [
            (self.chunk_id, 0.95),
            (chunk_id_2, 0.87),
            (chunk_id_3, 0.82)
        ]
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        mock_index.search.return_value = search_results
        self.mock_chunks.get.side_effect = [self.mock_chunk, chunk_2, chunk_3]
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding, k=3)
        
        # Assert
        assert len(result) == 3
        
        # Check first result (highest score)
        assert result[0]["chunk_id"] == self.chunk_id
        assert result[0]["score"] == 0.95
        assert result[0]["text"] == "Test chunk text"
        assert result[0]["metadata"]["page_number"] == 1
        
        # Check second result
        assert result[1]["chunk_id"] == chunk_id_2
        assert result[1]["score"] == 0.87
        assert result[1]["text"] == "Second chunk text"
        assert result[1]["metadata"]["page_number"] == 2
        
        # Check third result
        assert result[2]["chunk_id"] == chunk_id_3
        assert result[2]["score"] == 0.82
        assert result[2]["text"] == "Third chunk text"
        assert result[2]["metadata"]["page_number"] == 3
        
        mock_index.search.assert_called_once_with(self.query_embedding, 3, "cosine")
        assert self.mock_chunks.get.call_count == 3
    
    def test_query_chunk_metadata_handling(self):
        """Test search query with different chunk metadata scenarios"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        
        # Chunk with minimal metadata
        chunk_minimal = Chunk(
            id=self.chunk_id,
            library_id=self.library_id,
            document_id=self.document_id,
            text="Minimal chunk",
            position=0,
            embedding=[0.1, 0.2, 0.3] * 42 + [0.1, 0.2],
            metadata=ChunkMetadata()  # Empty metadata
        )
        
        search_results = [(self.chunk_id, 0.95)]
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        mock_index.search.return_value = search_results
        self.mock_chunks.get.return_value = chunk_minimal
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding)
        
        # Assert
        assert len(result) == 1
        assert result[0]["chunk_id"] == self.chunk_id
        assert result[0]["text"] == "Minimal chunk"
        # Check that metadata is a dict with default values (not empty)
        assert isinstance(result[0]["metadata"], dict)
        assert "page_number" in result[0]["metadata"]
        assert "token_count" in result[0]["metadata"]
    
    def test_query_different_library_embedding_dims(self):
        """Test search query with different library embedding dimensions"""
        # Arrange
        library_256d = Library(
            id=self.library_id,
            name="256D Library",
            embedding_dim=256,
            index_config=IndexType(type="flat"),
            metadata={}
        )
        
        query_256d = [0.1, 0.2, 0.3] * 85 + [0.1]  # 256 dimensions
        
        self.mock_libs.get.return_value = library_256d
        
        # Act
        with pytest.raises(ValidationError, match="Embedding dim mismatch: got 128, expected 256"):
            self.service.query(self.library_id, self.query_embedding)  # 128D query on 256D library
    
    def test_query_lock_management(self):
        """Test that read lock is properly acquired and released"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        search_results = [(self.chunk_id, 0.95)]
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        mock_index.search.return_value = search_results
        self.mock_chunks.get.return_value = self.mock_chunk
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding)
        
        # Assert
        assert len(result) == 1
        self.mock_locks.lock_for_library.assert_called_once_with(self.library_id)
        mock_lock.acquire_read.assert_called_once()
        mock_lock.release_read.assert_called_once()
    
    def test_query_result_structure(self):
        """Test that query results have the correct structure"""
        # Arrange
        mock_lock = Mock()
        mock_index = Mock()
        search_results = [(self.chunk_id, 0.95)]
        
        self.mock_libs.get.return_value = self.mock_library
        self.mock_locks.lock_for_library.return_value = mock_lock
        self.mock_indexes.get.return_value = mock_index
        mock_index.search.return_value = search_results
        self.mock_chunks.get.return_value = self.mock_chunk
        
        # Act
        result = self.service.query(self.library_id, self.query_embedding)
        
        # Assert
        assert len(result) == 1
        result_item = result[0]
        
        # Check all required fields are present
        assert "chunk_id" in result_item
        assert "document_id" in result_item
        assert "score" in result_item
        assert "text" in result_item
        assert "metadata" in result_item
        
        # Check field types
        assert isinstance(result_item["chunk_id"], type(self.chunk_id))
        assert isinstance(result_item["document_id"], type(self.document_id))
        assert isinstance(result_item["score"], float)
        assert isinstance(result_item["text"], str)
        assert isinstance(result_item["metadata"], dict)