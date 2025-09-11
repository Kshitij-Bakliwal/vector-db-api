"""
Unit tests for Document repository
"""

import pytest
from uuid import uuid4
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.repos.documents import DocumentRepo
from vector_db_api.models.entities import Document
from vector_db_api.models.metadata import DocumentMetadata


class TestDocumentRepo:
    """Test cases for Document repository"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.repo = DocumentRepo()
        self.library_id = uuid4()
        self.test_document = Document(
            library_id=self.library_id,
            metadata=DocumentMetadata(
                title="Test Document",
                summary="A test document for unit tests"
            )
        )
    
    def test_add_document(self):
        """Test adding a document to the repository"""
        # Act
        self.repo.add(self.test_document)
        
        # Assert
        assert len(self.repo.documents) == 1
        assert self.test_document.id in self.repo.documents
    
    def test_get_document_by_id(self):
        """Test retrieving a document by ID"""
        # Arrange
        self.repo.add(self.test_document)
        
        # Act
        retrieved_document = self.repo.get(self.test_document.id)
        
        # Assert
        assert retrieved_document is not None
        assert retrieved_document.id == self.test_document.id
        assert retrieved_document.library_id == self.test_document.library_id
        assert retrieved_document.metadata.title == "Test Document"
    
    def test_get_nonexistent_document(self):
        """Test retrieving a document that doesn't exist"""
        # Act
        retrieved_document = self.repo.get(uuid4())
        
        # Assert
        assert retrieved_document is None
    
    def test_list_documents_by_library(self):
        """Test listing documents by library"""
        # Arrange
        document2 = Document(
            library_id=self.library_id,
            metadata=DocumentMetadata(title="Test Document 2")
        )
        self.repo.add(self.test_document)
        self.repo.add(document2)
        
        # Act
        documents = self.repo.list_by_library(self.library_id)
        
        # Assert
        assert len(documents) == 2
        assert self.test_document in documents
        assert document2 in documents
    
    def test_list_documents_by_specific_library(self):
        """Test listing documents by specific library ID"""
        # Arrange
        other_library_id = uuid4()
        document2 = Document(
            library_id=other_library_id,
            metadata=DocumentMetadata(title="Other Library Document")
        )
        self.repo.add(self.test_document)
        self.repo.add(document2)
        
        # Act
        library_documents = self.repo.list_by_library(self.library_id)
        
        # Assert
        assert len(library_documents) == 1
        assert self.test_document in library_documents
        assert document2 not in library_documents
    
    def test_list_documents_by_nonexistent_library(self):
        """Test listing documents for a library that doesn't exist"""
        # Act
        documents = self.repo.list_by_library(uuid4())
        
        # Assert
        assert len(documents) == 0
    
    def test_update_document(self):
        """Test updating a document using update_on_version"""
        # Arrange
        self.repo.add(self.test_document)
        original_version = self.test_document.version
        
        # Act
        self.test_document.metadata.title = "Updated Document Title"
        success = self.repo.update_on_version(self.test_document, original_version)
        
        # Assert
        assert success is True
        updated_document = self.repo.get(self.test_document.id)
        assert updated_document.metadata.title == "Updated Document Title"
        assert updated_document.version == original_version + 1
    
    def test_update_nonexistent_document(self):
        """Test updating a document that doesn't exist"""
        # Act
        success = self.repo.update_on_version(self.test_document, 1)
        
        # Assert
        assert success is False
    
    def test_delete_document(self):
        """Test deleting a document"""
        # Arrange
        self.repo.add(self.test_document)
        
        # Act
        deleted = self.repo.delete(self.test_document.id)
        
        # Assert
        assert deleted is True
        assert len(self.repo.documents) == 0
        assert self.repo.get(self.test_document.id) is None
    
    def test_delete_nonexistent_document(self):
        """Test deleting a document that doesn't exist"""
        # Act
        deleted = self.repo.delete(uuid4())
        
        # Assert
        assert deleted is False
    
    def test_document_version_increment(self):
        """Test that document version increments on update"""
        # Arrange
        self.repo.add(self.test_document)
        original_version = self.test_document.version
        
        # Act
        self.test_document.metadata.summary = "Updated summary"
        success = self.repo.update_on_version(self.test_document, original_version)
        
        # Assert
        assert success is True
        updated_document = self.repo.get(self.test_document.id)
        assert updated_document.version == original_version + 1
    
    def test_document_timestamps(self):
        """Test that timestamps are properly managed"""
        # Arrange - create a new document for this test
        test_doc = Document(
            library_id=self.library_id,
            metadata=DocumentMetadata(
                title="Timestamp Test Document",
                summary="A test document for timestamps"
            )
        )
        
        # Store original timestamps (these will be different from what the repo sets)
        original_created_at = test_doc.created_at
        original_updated_at = test_doc.updated_at
        
        # Act - repository will set its own timestamps
        before_add = datetime.utcnow()
        self.repo.add(test_doc)
        after_add = datetime.utcnow()
        
        # Assert - repository should have set new timestamps
        assert before_add <= test_doc.created_at <= after_add
        assert before_add <= test_doc.updated_at <= after_add
        # created_at and updated_at should be the same when document is added
        assert test_doc.created_at == test_doc.updated_at
        
        # Test that repository preserves timestamps correctly
        retrieved_document = self.repo.get(test_doc.id)
        assert retrieved_document.created_at == test_doc.created_at
        assert retrieved_document.updated_at == test_doc.updated_at
        assert retrieved_document.created_at == retrieved_document.updated_at
        
        # Test update - version should increment and updated_at should be updated
        test_doc.metadata.title = "Updated Title"
        current_version = test_doc.version  # Store the current version before update
        # Store the updated_at timestamp before the update
        updated_at_before_update = test_doc.updated_at
        before_update = datetime.utcnow()
        success = self.repo.update_on_version(test_doc, current_version)
        after_update = datetime.utcnow()
        
        assert success is True
        updated_document = self.repo.get(test_doc.id)
        # Version should increment
        assert updated_document.version == current_version + 1
        # created_at should remain the same as when it was added
        assert updated_document.created_at == test_doc.created_at
        # updated_at should be newer than the original and within our time window
        assert updated_document.updated_at > updated_at_before_update
        assert before_update <= updated_document.updated_at <= after_update
    
    def test_document_chunk_ids_management(self):
        """Test that chunk IDs are properly managed"""
        # Arrange
        self.repo.add(self.test_document)
        chunk_id1 = uuid4()
        chunk_id2 = uuid4()
        
        # Act
        self.test_document.chunk_ids.append(chunk_id1)
        self.test_document.chunk_ids.append(chunk_id2)
        success = self.repo.update_on_version(self.test_document, self.test_document.version)
        
        # Assert
        assert success is True
        updated_document = self.repo.get(self.test_document.id)
        assert len(updated_document.chunk_ids) == 2
        assert chunk_id1 in updated_document.chunk_ids
        assert chunk_id2 in updated_document.chunk_ids
    
    def test_update_on_version(self):
        """Test optimistic concurrency control with version"""
        # Arrange
        self.repo.add(self.test_document)
        original_version = self.test_document.version
        
        # Act
        self.test_document.metadata.title = "Updated Title"
        success = self.repo.update_on_version(self.test_document, original_version)
        
        # Assert
        assert success is True
        updated_document = self.repo.get(self.test_document.id)
        assert updated_document.metadata.title == "Updated Title"
        assert updated_document.version == original_version + 1
    
    def test_update_on_wrong_version(self):
        """Test that update fails with wrong version"""
        # Arrange
        self.repo.add(self.test_document)
        wrong_version = self.test_document.version + 1
        
        # Act
        self.test_document.metadata.title = "Updated Title"
        success = self.repo.update_on_version(self.test_document, wrong_version)
        
        # Assert
        assert success is False
