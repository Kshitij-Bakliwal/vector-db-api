"""
Pytest configuration and shared fixtures
"""

import pytest
import sys
import os
from uuid import uuid4

# Add src to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vector_db_api.models.entities import Library, Document, Chunk
from vector_db_api.models.metadata import LibraryMetadata, DocumentMetadata, ChunkMetadata
from vector_db_api.models.indexing import IndexType


@pytest.fixture
def sample_library():
    """Create a sample library for testing"""
    return Library(
        name="Test Library",
        embedding_dim=128,
        metadata=LibraryMetadata(description="Test library for unit tests"),
        index_config=IndexType(type="flat")
    )


@pytest.fixture
def sample_document(sample_library):
    """Create a sample document for testing"""
    return Document(
        library_id=sample_library.id,
        metadata=DocumentMetadata(
            title="Test Document",
            summary="A test document for unit tests"
        )
    )


@pytest.fixture
def sample_chunk(sample_library, sample_document):
    """Create a sample chunk for testing"""
    return Chunk(
        library_id=sample_library.id,
        document_id=sample_document.id,
        text="This is a test chunk for unit testing.",
        position=0,
        embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        metadata=ChunkMetadata(
            token_count=8,
            tags=["test", "unit"]
        )
    )


@pytest.fixture
def sample_embedding_128d():
    """Create a sample 128-dimensional embedding"""
    return [0.1] * 128


@pytest.fixture
def sample_embedding_10d():
    """Create a sample 10-dimensional embedding"""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


@pytest.fixture
def mock_repositories():
    """Create mock repositories for testing"""
    from unittest.mock import Mock
    
    return {
        'libraries': Mock(),
        'documents': Mock(),
        'chunks': Mock(),
        'locks': Mock(),
        'indexes': Mock()
    }
