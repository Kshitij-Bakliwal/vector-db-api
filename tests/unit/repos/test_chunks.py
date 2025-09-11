"""
Unit tests for Chunk repository
"""

import pytest
from uuid import uuid4
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.repos.chunks import ChunkRepo
from vector_db_api.models.entities import Chunk
from vector_db_api.models.metadata import ChunkMetadata


class TestChunkRepo:
    """Test cases for Chunk repository"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.repo = ChunkRepo()
        self.library_id = uuid4()
        self.document_id = uuid4()
        self.test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        self.test_chunk = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="This is a test chunk for unit testing.",
            position=0,
            embedding=self.test_embedding,
            metadata=ChunkMetadata(
                token_count=8,
                tags=["test", "unit"]
            )
        )
    
    def test_add_chunk(self):
        """Test adding a chunk to the repository"""
        # Act
        self.repo.add(self.test_chunk)
        
        # Assert
        assert len(self.repo.chunks) == 1
        assert self.test_chunk.id in self.repo.chunks
    
    def test_get_chunk_by_id(self):
        """Test retrieving a chunk by ID"""
        # Arrange
        self.repo.add(self.test_chunk)
        
        # Act
        retrieved_chunk = self.repo.get(self.test_chunk.id)
        
        # Assert
        assert retrieved_chunk is not None
        assert retrieved_chunk.id == self.test_chunk.id
        assert retrieved_chunk.library_id == self.test_chunk.library_id
        assert retrieved_chunk.document_id == self.test_chunk.document_id
        assert retrieved_chunk.text == self.test_chunk.text
        assert retrieved_chunk.embedding == self.test_embedding
    
    def test_get_nonexistent_chunk(self):
        """Test retrieving a chunk that doesn't exist"""
        # Act
        retrieved_chunk = self.repo.get(uuid4())
        
        # Assert
        assert retrieved_chunk is None
    
    def test_list_chunks_by_library(self):
        """Test listing chunks by library"""
        # Arrange
        chunk2 = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="Second test chunk",
            position=1,
            embedding=[0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self.repo.add(self.test_chunk)
        self.repo.add(chunk2)
        
        # Act
        chunks = self.repo.list_by_library(self.library_id)
        
        # Assert
        assert len(chunks) == 2
        assert self.test_chunk in chunks
        assert chunk2 in chunks
    
    def test_list_chunks_by_specific_library(self):
        """Test listing chunks by specific library ID"""
        # Arrange
        other_library_id = uuid4()
        chunk2 = Chunk(
            library_id=other_library_id,
            document_id=self.document_id,
            text="Other library chunk",
            position=0,
            embedding=[0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self.repo.add(self.test_chunk)
        self.repo.add(chunk2)
        
        # Act
        library_chunks = self.repo.list_by_library(self.library_id)
        
        # Assert
        assert len(library_chunks) == 1
        assert self.test_chunk in library_chunks
        assert chunk2 not in library_chunks
    
    def test_list_chunks_by_document(self):
        """Test listing chunks by document ID"""
        # Arrange
        other_document_id = uuid4()
        chunk2 = Chunk(
            library_id=self.library_id,
            document_id=other_document_id,
            text="Other document chunk",
            position=0,
            embedding=[0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self.repo.add(self.test_chunk)
        self.repo.add(chunk2)
        
        # Act
        document_chunks = self.repo.list_by_document(self.document_id)
        
        # Assert
        assert len(document_chunks) == 1
        assert self.test_chunk in document_chunks
        assert chunk2 not in document_chunks
    
    def test_list_chunks_by_nonexistent_library(self):
        """Test listing chunks for a library that doesn't exist"""
        # Act
        chunks = self.repo.list_by_library(uuid4())
        
        # Assert
        assert len(chunks) == 0
    
    def test_list_chunks_by_nonexistent_document(self):
        """Test listing chunks for a document that doesn't exist"""
        # Act
        chunks = self.repo.list_by_document(uuid4())
        
        # Assert
        assert len(chunks) == 0
    
    def test_update_chunk(self):
        """Test updating a chunk using update_on_version"""
        # Arrange
        self.repo.add(self.test_chunk)
        original_version = self.test_chunk.version
        
        # Act
        self.test_chunk.text = "Updated chunk text"
        self.test_chunk.position = 1
        success = self.repo.update_on_version(self.test_chunk, original_version)
        
        # Assert
        assert success is True
        updated_chunk = self.repo.get(self.test_chunk.id)
        assert updated_chunk.text == "Updated chunk text"
        assert updated_chunk.position == 1
        assert updated_chunk.version == original_version + 1
    
    def test_update_nonexistent_chunk(self):
        """Test updating a chunk that doesn't exist"""
        # Act
        success = self.repo.update_on_version(self.test_chunk, 1)
        
        # Assert
        assert success is False
    
    def test_delete_chunk(self):
        """Test deleting a chunk"""
        # Arrange
        self.repo.add(self.test_chunk)
        
        # Act
        deleted = self.repo.delete(self.test_chunk.id)
        
        # Assert
        assert deleted is True
        assert len(self.repo.chunks) == 0
        assert self.repo.get(self.test_chunk.id) is None
    
    def test_delete_nonexistent_chunk(self):
        """Test deleting a chunk that doesn't exist"""
        # Act
        deleted = self.repo.delete(uuid4())
        
        # Assert
        assert deleted is False
    
    def test_chunk_version_increment(self):
        """Test that chunk version increments on update"""
        # Arrange
        self.repo.add(self.test_chunk)
        original_version = self.test_chunk.version
        
        # Act
        self.test_chunk.text = "Updated text"
        success = self.repo.update_on_version(self.test_chunk, original_version)
        
        # Assert
        assert success is True
        updated_chunk = self.repo.get(self.test_chunk.id)
        assert updated_chunk.version == original_version + 1
    
    def test_chunk_timestamps(self):
        """Test that timestamps are properly managed"""
        # Arrange - create a new chunk for this test
        test_chunk = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="Timestamp test chunk",
            position=0,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
            metadata=ChunkMetadata(
                token_count=4,
                tags=["timestamp", "test"]
            )
        )
        
        # Store original timestamps (these will be different from what the repo sets)
        original_created_at = test_chunk.created_at
        original_updated_at = test_chunk.updated_at
        
        # Act - repository will set its own timestamps
        before_add = datetime.utcnow()
        self.repo.add(test_chunk)
        after_add = datetime.utcnow()
        
        # Assert - repository should have set new timestamps
        assert before_add <= test_chunk.created_at <= after_add
        assert before_add <= test_chunk.updated_at <= after_add
        # created_at and updated_at should be the same when chunk is added
        assert test_chunk.created_at == test_chunk.updated_at
        
        # Test that repository preserves timestamps correctly
        retrieved_chunk = self.repo.get(test_chunk.id)
        assert retrieved_chunk.created_at == test_chunk.created_at
        assert retrieved_chunk.updated_at == test_chunk.updated_at
        assert retrieved_chunk.created_at == retrieved_chunk.updated_at
        
        # Test update - version should increment and updated_at should be updated
        test_chunk.text = "Updated text"
        current_version = test_chunk.version  # Store the current version before update
        # Store the updated_at timestamp before the update
        updated_at_before_update = test_chunk.updated_at
        before_update = datetime.utcnow()
        success = self.repo.update_on_version(test_chunk, current_version)
        after_update = datetime.utcnow()
        
        assert success is True
        updated_chunk = self.repo.get(test_chunk.id)
        # Version should increment
        assert updated_chunk.version == current_version + 1
        # created_at should remain the same as when it was added
        assert updated_chunk.created_at == test_chunk.created_at
        # updated_at should be newer than the original and within our time window
        assert updated_chunk.updated_at > updated_at_before_update
        assert before_update <= updated_chunk.updated_at <= after_update
    
    def test_chunk_embedding_management(self):
        """Test that embeddings are properly stored and retrieved"""
        # Arrange
        new_embedding = [0.9, 0.8, 0.7, 0.6, 0.5]
        self.repo.add(self.test_chunk)
        
        # Act
        self.test_chunk.embedding = new_embedding
        success = self.repo.update_on_version(self.test_chunk, self.test_chunk.version)
        
        # Assert
        assert success is True
        updated_chunk = self.repo.get(self.test_chunk.id)
        assert updated_chunk.embedding == new_embedding
        assert len(updated_chunk.embedding) == 5
    
    def test_chunk_metadata_management(self):
        """Test that metadata is properly stored and retrieved"""
        # Arrange
        self.repo.add(self.test_chunk)
        
        # Act
        self.test_chunk.metadata.token_count = 15
        self.test_chunk.metadata.tags.append("updated")
        success = self.repo.update_on_version(self.test_chunk, self.test_chunk.version)
        
        # Assert
        assert success is True
        updated_chunk = self.repo.get(self.test_chunk.id)
        assert updated_chunk.metadata.token_count == 15
        assert "updated" in updated_chunk.metadata.tags
        assert len(updated_chunk.metadata.tags) == 3  # original 2 + 1 new
    
    def test_update_on_version(self):
        """Test optimistic concurrency control with version"""
        # Arrange
        self.repo.add(self.test_chunk)
        original_version = self.test_chunk.version
        
        # Act
        self.test_chunk.text = "Updated text"
        success = self.repo.update_on_version(self.test_chunk, original_version)
        
        # Assert
        assert success is True
        updated_chunk = self.repo.get(self.test_chunk.id)
        assert updated_chunk.text == "Updated text"
        assert updated_chunk.version == original_version + 1
    
    def test_update_on_wrong_version(self):
        """Test that update fails with wrong version"""
        # Arrange
        self.repo.add(self.test_chunk)
        wrong_version = self.test_chunk.version + 1
        
        # Act
        self.test_chunk.text = "Updated text"
        success = self.repo.update_on_version(self.test_chunk, wrong_version)
        
        # Assert
        assert success is False
    
    def test_bulk_operations(self):
        """Test bulk operations on chunks"""
        # Arrange
        chunk2 = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="Second chunk",
            position=1,
            embedding=[0.6, 0.7, 0.8, 0.9, 1.0]
        )
        chunk3 = Chunk(
            library_id=self.library_id,
            document_id=self.document_id,
            text="Third chunk",
            position=2,
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
        )
        chunks_to_add = [self.test_chunk, chunk2, chunk3]
        
        # Act
        self.repo.add_bulk(chunks_to_add)
        
        # Assert
        assert len(self.repo.chunks) == 3
        all_chunks = self.repo.list_by_library(self.library_id)
        assert len(all_chunks) == 3
        
        # Verify all chunks are properly indexed by document
        document_chunks = self.repo.list_by_document(self.document_id)
        assert len(document_chunks) == 3
        
        # Verify individual chunks can be retrieved
        assert self.repo.get(self.test_chunk.id) is not None
        assert self.repo.get(chunk2.id) is not None
        assert self.repo.get(chunk3.id) is not None
