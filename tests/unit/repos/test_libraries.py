"""
Unit tests for Library repository
"""

import pytest
from uuid import uuid4
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.repos.libraries import LibraryRepo
from vector_db_api.models.entities import Library
from vector_db_api.models.metadata import LibraryMetadata
from vector_db_api.models.indexing import IndexType


class TestLibraryRepo:
    """Test cases for Library repository"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.repo = LibraryRepo()
        self.test_library = Library(
            name="Test Library",
            embedding_dim=128,
            metadata=LibraryMetadata(description="Test library for unit tests"),
            index_config=IndexType(type="flat")
        )
    
    def test_add_library(self):
        """Test adding a library to the repository"""
        # Act
        self.repo.add(self.test_library)
        
        # Assert
        assert len(self.repo.libraries) == 1
        assert self.test_library.id in self.repo.libraries
    
    def test_get_library_by_id(self):
        """Test retrieving a library by ID"""
        # Arrange
        self.repo.add(self.test_library)
        
        # Act
        retrieved_library = self.repo.get(self.test_library.id)
        
        # Assert
        assert retrieved_library is not None
        assert retrieved_library.id == self.test_library.id
        assert retrieved_library.name == self.test_library.name
        assert retrieved_library.embedding_dim == self.test_library.embedding_dim
    
    def test_get_nonexistent_library(self):
        """Test retrieving a library that doesn't exist"""
        # Act
        retrieved_library = self.repo.get(uuid4())
        
        # Assert
        assert retrieved_library is None
    
    def test_list_libraries(self):
        """Test listing all libraries"""
        # Arrange
        library2 = Library(
            name="Test Library 2",
            embedding_dim=256,
            metadata=LibraryMetadata(description="Second test library")
        )
        self.repo.add(self.test_library)
        self.repo.add(library2)
        
        # Act
        libraries = self.repo.list()
        
        # Assert
        assert len(libraries) == 2
        assert self.test_library in libraries
        assert library2 in libraries
    
    def test_list_empty_repository(self):
        """Test listing libraries from empty repository"""
        # Act
        libraries = self.repo.list()
        
        # Assert
        assert len(libraries) == 0
    
    def test_update_library(self):
        """Test updating a library using update_on_version"""
        # Arrange
        self.repo.add(self.test_library)
        original_version = self.test_library.version
        
        # Act
        self.test_library.name = "Updated Library Name"
        self.test_library.embedding_dim = 256
        success = self.repo.update_on_version(self.test_library, original_version)
        
        # Assert
        assert success is True
        updated_library = self.repo.get(self.test_library.id)
        assert updated_library.name == "Updated Library Name"
        assert updated_library.embedding_dim == 256
        assert updated_library.version == original_version + 1
    
    def test_update_nonexistent_library(self):
        """Test updating a library that doesn't exist"""
        # Act
        success = self.repo.update_on_version(self.test_library, 1)
        
        # Assert
        assert success is False
    
    def test_delete_library(self):
        """Test deleting a library"""
        # Arrange
        self.repo.add(self.test_library)
        
        # Act
        deleted = self.repo.delete(self.test_library.id)
        
        # Assert
        assert deleted is True
        assert len(self.repo.libraries) == 0
        assert self.repo.get(self.test_library.id) is None
    
    def test_delete_nonexistent_library(self):
        """Test deleting a library that doesn't exist"""
        # Act
        deleted = self.repo.delete(uuid4())
        
        # Assert
        assert deleted is False
    
    def test_library_version_increment(self):
        """Test that library version increments on update"""
        # Arrange
        self.repo.add(self.test_library)
        original_version = self.test_library.version
        
        # Act
        self.test_library.name = "Updated Name"
        success = self.repo.update_on_version(self.test_library, original_version)
        
        # Assert
        assert success is True
        updated_library = self.repo.get(self.test_library.id)
        assert updated_library.version == original_version + 1
    
    def test_library_timestamps(self):
        """Test that timestamps are properly managed"""
        # Arrange - create a new library for this test
        test_lib = Library(
            name="Timestamp Test Library",
            embedding_dim=128,
            metadata=LibraryMetadata(description="Test library for timestamps"),
            index_config=IndexType(type="flat")
        )
        
        # Store original timestamps (these will be different from what the repo sets)
        original_created_at = test_lib.created_at
        original_updated_at = test_lib.updated_at
        
        # Act - repository will set its own timestamps
        before_add = datetime.utcnow()
        self.repo.add(test_lib)
        after_add = datetime.utcnow()
        
        # Assert - repository should have set new timestamps
        assert before_add <= test_lib.created_at <= after_add
        assert before_add <= test_lib.updated_at <= after_add
        # created_at and updated_at should be the same when library is added
        assert test_lib.created_at == test_lib.updated_at
        
        # Test that repository preserves timestamps correctly
        retrieved_library = self.repo.get(test_lib.id)
        assert retrieved_library.created_at == test_lib.created_at
        assert retrieved_library.updated_at == test_lib.updated_at
        assert retrieved_library.created_at == retrieved_library.updated_at
        
        # Test update - version should increment and updated_at should be updated
        test_lib.name = "Updated"
        current_version = test_lib.version  # Store the current version before update
        # Store the updated_at timestamp before the update
        updated_at_before_update = test_lib.updated_at
        before_update = datetime.utcnow()
        success = self.repo.update_on_version(test_lib, current_version)
        after_update = datetime.utcnow()
        
        assert success is True
        updated_library = self.repo.get(test_lib.id)
        # Version should increment
        assert updated_library.version == current_version + 1
        # created_at should remain the same as when it was added
        assert updated_library.created_at == test_lib.created_at
        # updated_at should be newer than the original and within our time window
        assert updated_library.updated_at > updated_at_before_update
        assert before_update <= updated_library.updated_at <= after_update
