"""
Unit tests for IndexRegistry
"""

import pytest
from uuid import uuid4
from datetime import datetime
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.indexing.registry import IndexRegistry
from vector_db_api.indexing.flat import FlatIndex
from vector_db_api.indexing.lsh import LSHIndex
from vector_db_api.indexing.ivf import IVFIndex
from vector_db_api.models.indexing import IndexType
from vector_db_api.models.entities import Library


class TestIndexRegistry:
    """Test cases for IndexRegistry"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.registry = IndexRegistry()
        self.library_id = uuid4()
        self.embedding_dim = 128
        
        # Create sample libraries with different index types
        self.flat_library = Library(
            id=self.library_id,
            name="Flat Library",
            embedding_dim=self.embedding_dim,
            index_config=IndexType(type="flat"),
            metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1
        )
        
        self.lsh_library = Library(
            id=uuid4(),
            name="LSH Library",
            embedding_dim=self.embedding_dim,
            index_config=IndexType(
                type="lsh",
                lsh_num_tables=8,
                lsh_hyperplanes_per_table=16
            ),
            metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1
        )
        
        self.ivf_library = Library(
            id=uuid4(),
            name="IVF Library",
            embedding_dim=self.embedding_dim,
            index_config=IndexType(
                type="ivf",
                ivf_num_centroids=64,
                ivf_nprobe=4
            ),
            metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            version=1
        )
    
    def test_registry_initialization(self):
        """Test registry initialization"""
        # Assert
        assert len(self.registry.registry) == 0
        assert self.registry.lock is not None
    
    def test_add_flat_index(self):
        """Test adding a flat index"""
        # Act
        index = self.registry.get_or_create(self.flat_library)
        
        # Assert
        assert isinstance(index, FlatIndex)
        assert self.flat_library.id in self.registry.registry
        assert self.registry.registry[self.flat_library.id] == index
    
    def test_add_lsh_index(self):
        """Test adding an LSH index"""
        # Act
        index = self.registry.get_or_create(self.lsh_library)
        
        # Assert
        assert isinstance(index, LSHIndex)
        assert index.dim == self.embedding_dim
        assert index.L == 8  # lsh_num_tables
        assert index.H == 16  # lsh_hyperplanes_per_table
        assert self.lsh_library.id in self.registry.registry
    
    def test_add_ivf_index(self):
        """Test adding an IVF index"""
        # Act
        index = self.registry.get_or_create(self.ivf_library)
        
        # Assert
        assert isinstance(index, IVFIndex)
        assert index.dim == self.embedding_dim
        assert index.k == 64  # ivf_num_centroids
        assert index.nprobe == 4  # ivf_nprobe
        assert self.ivf_library.id in self.registry.registry
    
    def test_add_existing_library(self):
        """Test adding an index for an existing library"""
        # Arrange
        index1 = self.registry.get_or_create(self.flat_library)
        
        # Act
        index2 = self.registry.get_or_create(self.flat_library)
        
        # Assert
        assert index1 == index2  # Should return the same index
        assert len(self.registry.registry) == 1
    
    def test_get_existing_index(self):
        """Test getting an existing index"""
        # Arrange
        added_index = self.registry.get_or_create(self.flat_library)
        
        # Act
        retrieved_index = self.registry.get(self.flat_library.id)
        
        # Assert
        assert retrieved_index == added_index
        assert retrieved_index is not None
    
    def test_get_nonexistent_index(self):
        """Test getting a non-existent index"""
        # Act
        index = self.registry.get(self.library_id)
        
        # Assert
        assert index is None
    
    def test_swap_index(self):
        """Test swapping an index"""
        # Arrange
        old_index = self.registry.get_or_create(self.flat_library)
        new_index = FlatIndex()
        
        # Act
        self.registry.swap(self.flat_library.id, new_index)
        
        # Assert
        retrieved_index = self.registry.get(self.flat_library.id)
        assert retrieved_index == new_index
        assert retrieved_index != old_index
    
    def test_swap_nonexistent_library(self):
        """Test swapping index for non-existent library"""
        # Arrange
        new_index = FlatIndex()
        
        # Act
        self.registry.swap(self.library_id, new_index)
        
        # Assert
        retrieved_index = self.registry.get(self.library_id)
        assert retrieved_index == new_index
    
    def test_remove_existing_index(self):
        """Test removing an existing index"""
        # Arrange
        self.registry.get_or_create(self.flat_library)
        
        # Act
        self.registry.remove(self.flat_library.id)
        
        # Assert
        assert self.flat_library.id not in self.registry.registry
        assert self.registry.get(self.flat_library.id) is None
    
    def test_remove_nonexistent_index(self):
        """Test removing a non-existent index"""
        # Act
        self.registry.remove(self.library_id)  # Should not raise error
        
        # Assert
        assert self.library_id not in self.registry.registry
    
    def test_create_index_flat_type(self):
        """Test create_index method for flat type"""
        # Act
        index = self.registry.create_index(
            IndexType(type="flat"),
            self.embedding_dim
        )
        
        # Assert
        assert isinstance(index, FlatIndex)
    
    def test_create_index_lsh_type(self):
        """Test create_index method for LSH type"""
        # Act
        index = self.registry.create_index(
            IndexType(
                type="lsh",
                lsh_num_tables=4,
                lsh_hyperplanes_per_table=8
            ),
            self.embedding_dim
        )
        
        # Assert
        assert isinstance(index, LSHIndex)
        assert index.dim == self.embedding_dim
        assert index.L == 4
        assert index.H == 8
    
    def test_create_index_ivf_type(self):
        """Test create_index method for IVF type"""
        # Act
        index = self.registry.create_index(
            IndexType(
                type="ivf",
                ivf_num_centroids=32,
                ivf_nprobe=2
            ),
            self.embedding_dim
        )
        
        # Assert
        assert isinstance(index, IVFIndex)
        assert index.dim == self.embedding_dim
        assert index.k == 32
        assert index.nprobe == 2
    
    def test_create_index_unsupported_type(self):
        """Test create_index method with unsupported type"""
        # Arrange - create a mock config with unsupported type
        class UnsupportedConfig:
            type = "unsupported"
        
        unsupported_config = UnsupportedConfig()
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported index type"):
            self.registry.create_index(unsupported_config, self.embedding_dim)
    
    def test_create_index_default_parameters(self):
        """Test create_index with default parameters"""
        # Act
        lsh_index = self.registry.create_index(
            IndexType(type="lsh"),  # No explicit parameters
            self.embedding_dim
        )
        
        ivf_index = self.registry.create_index(
            IndexType(type="ivf"),  # No explicit parameters
            self.embedding_dim
        )
        
        # Assert
        assert isinstance(lsh_index, LSHIndex)
        assert lsh_index.L == 8  # Default lsh_num_tables
        assert lsh_index.H == 16  # Default lsh_hyperplanes_per_table
        
        assert isinstance(ivf_index, IVFIndex)
        assert ivf_index.k == 64  # Default ivf_num_centroids
        assert ivf_index.nprobe == 4  # Default ivf_nprobe
    
    def test_multiple_libraries(self):
        """Test registry with multiple libraries"""
        # Act
        flat_index = self.registry.get_or_create(self.flat_library)
        lsh_index = self.registry.get_or_create(self.lsh_library)
        ivf_index = self.registry.get_or_create(self.ivf_library)
        
        # Assert
        assert len(self.registry.registry) == 3
        assert self.registry.get(self.flat_library.id) == flat_index
        assert self.registry.get(self.lsh_library.id) == lsh_index
        assert self.registry.get(self.ivf_library.id) == ivf_index
    
    def test_registry_thread_safety(self):
        """Test that registry operations are thread-safe"""
        import threading
        import time
        
        # Arrange
        results = []
        errors = []
        
        def add_library(library):
            try:
                index = self.registry.get_or_create(library)
                results.append((library.id, index))
            except Exception as e:
                errors.append(e)
        
        # Act
        threads = []
        for i in range(10):
            library = Library(
                id=uuid4(),
                name=f"Library {i}",
                embedding_dim=self.embedding_dim,
                index_config=IndexType(type="flat"),
                metadata={},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                version=1
            )
            thread = threading.Thread(target=add_library, args=(library,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Assert
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == 10
        assert len(self.registry.registry) == 10
    
    def test_swap_and_remove_operations(self):
        """Test swap and remove operations together"""
        # Arrange
        original_index = self.registry.get_or_create(self.flat_library)
        new_index = LSHIndex(dim=self.embedding_dim, num_tables=4, hyperplanes_per_table=8)
        
        # Act
        self.registry.swap(self.flat_library.id, new_index)
        retrieved_index = self.registry.get(self.flat_library.id)
        
        # Assert
        assert retrieved_index == new_index
        assert retrieved_index != original_index
        
        # Act
        self.registry.remove(self.flat_library.id)
        
        # Assert
        assert self.registry.get(self.flat_library.id) is None
    
    def test_index_configuration_parameters(self):
        """Test different index configuration parameters"""
        # Test LSH with custom parameters
        lsh_config = IndexType(
            type="lsh",
            lsh_num_tables=16,
            lsh_hyperplanes_per_table=32
        )
        lsh_index = self.registry.create_index(lsh_config, self.embedding_dim)
        assert lsh_index.L == 16
        assert lsh_index.H == 32
        
        # Test IVF with custom parameters
        ivf_config = IndexType(
            type="ivf",
            ivf_num_centroids=128,
            ivf_nprobe=8
        )
        ivf_index = self.registry.create_index(ivf_config, self.embedding_dim)
        assert ivf_index.k == 128
        assert ivf_index.nprobe == 8
    
    def test_registry_cleanup(self):
        """Test registry cleanup operations"""
        # Arrange
        self.registry.get_or_create(self.flat_library)
        self.registry.get_or_create(self.lsh_library)
        self.registry.get_or_create(self.ivf_library)
        
        # Act
        self.registry.remove(self.flat_library.id)
        self.registry.remove(self.lsh_library.id)
        self.registry.remove(self.ivf_library.id)
        
        # Assert
        assert len(self.registry.registry) == 0
        assert self.registry.get(self.flat_library.id) is None
        assert self.registry.get(self.lsh_library.id) is None
        assert self.registry.get(self.ivf_library.id) is None
    
    def test_index_reuse_after_removal(self):
        """Test that index can be recreated after removal"""
        # Arrange
        original_index = self.registry.get_or_create(self.flat_library)
        
        # Act
        self.registry.remove(self.flat_library.id)
        new_index = self.registry.get_or_create(self.flat_library)
        
        # Assert
        assert new_index != original_index  # Should be a new instance
        assert isinstance(new_index, FlatIndex)
        assert self.registry.get(self.flat_library.id) == new_index
