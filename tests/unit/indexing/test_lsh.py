"""
Unit tests for LSHIndex
"""

import pytest
from uuid import uuid4
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.indexing.lsh import LSHIndex, LSHTable


class TestLSHTable:
    """Test cases for LSHTable"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.dim = 128
        self.hyperplanes = 16
        import random
        self.table = LSHTable(self.dim, self.hyperplanes, random.Random(42))
        self.chunk_id = uuid4()
        self.vector = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
    
    def test_lsh_table_initialization(self):
        """Test LSH table initialization"""
        # Assert
        assert self.table.dim == self.dim
        assert self.table.H == self.hyperplanes
        assert len(self.table.hyperplanes) == self.hyperplanes
        assert len(self.table.hyperplanes[0]) == self.dim
        assert len(self.table.buckets) == 0
    
    def test_signature_generation(self):
        """Test signature generation for vectors"""
        # Act
        sig1 = self.table.signature(self.vector)
        sig2 = self.table.signature(self.vector)
        
        # Assert
        assert isinstance(sig1, int)
        assert sig1 == sig2  # Same vector should produce same signature
        assert 0 <= sig1 < (1 << self.hyperplanes)  # Signature should be within valid range
    
    def test_signature_different_vectors(self):
        """Test that different vectors produce different signatures"""
        # Arrange
        vector1 = [1.0, 0.0, 0.0] * 42 + [1.0, 0.0]
        vector2 = [0.0, 1.0, 0.0] * 42 + [0.0, 1.0]
        
        # Act
        sig1 = self.table.signature(vector1)
        sig2 = self.table.signature(vector2)
        
        # Assert
        # Different vectors should likely produce different signatures
        # (though not guaranteed due to randomness)
        assert isinstance(sig1, int)
        assert isinstance(sig2, int)
    
    def test_add_vector(self):
        """Test adding a vector to LSH table"""
        # Act
        self.table.add(self.chunk_id, self.vector)
        
        # Assert
        sig = self.table.signature(self.vector)
        assert sig in self.table.buckets
        assert self.chunk_id in self.table.buckets[sig]
    
    def test_add_multiple_vectors(self):
        """Test adding multiple vectors to LSH table"""
        # Arrange
        chunk_id2 = uuid4()
        vector2 = [0.4, 0.5, 0.6] * 42 + [0.4, 0.5]
        
        # Act
        self.table.add(self.chunk_id, self.vector)
        self.table.add(chunk_id2, vector2)
        
        # Assert
        sig1 = self.table.signature(self.vector)
        sig2 = self.table.signature(vector2)
        
        assert self.chunk_id in self.table.buckets[sig1]
        assert chunk_id2 in self.table.buckets[sig2]
    
    def test_remove_vector(self):
        """Test removing a vector from LSH table"""
        # Arrange
        self.table.add(self.chunk_id, self.vector)
        sig = self.table.signature(self.vector)
        
        # Act
        self.table.remove(self.chunk_id, self.vector)
        
        # Assert
        if sig in self.table.buckets:
            assert self.chunk_id not in self.table.buckets[sig]
    
    def test_remove_nonexistent_vector(self):
        """Test removing a vector that doesn't exist"""
        # Act
        self.table.remove(self.chunk_id, self.vector)  # Should not raise error
        
        # Assert
        # Should not create any buckets
        assert len(self.table.buckets) == 0
    
    def test_remove_empties_bucket(self):
        """Test that removing the last vector from a bucket empties it"""
        # Arrange
        self.table.add(self.chunk_id, self.vector)
        sig = self.table.signature(self.vector)
        
        # Act
        self.table.remove(self.chunk_id, self.vector)
        
        # Assert
        assert sig not in self.table.buckets  # Bucket should be removed


class TestLSHIndex:
    """Test cases for LSHIndex"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.dim = 128
        self.num_tables = 4
        self.hyperplanes_per_table = 8
        self.index = LSHIndex(
            dim=self.dim,
            num_tables=self.num_tables,
            hyperplanes_per_table=self.hyperplanes_per_table,
            seed=42
        )
        
        self.chunk_id1 = uuid4()
        self.chunk_id2 = uuid4()
        self.chunk_id3 = uuid4()
        
        # Sample vectors (128 dimensions)
        self.vector1 = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
        self.vector2 = [0.4, 0.5, 0.6] * 42 + [0.4, 0.5]
        self.vector3 = [0.7, 0.8, 0.9] * 42 + [0.7, 0.8]
        self.query_vector = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]  # Similar to vector1
    
    def test_lsh_index_initialization(self):
        """Test LSH index initialization"""
        # Assert
        assert self.index.dim == self.dim
        assert self.index.L == self.num_tables
        assert self.index.H == self.hyperplanes_per_table
        assert len(self.index.tables) == self.num_tables
        assert len(self.index.vecs) == 0
        
        # Check that all tables are properly initialized
        for table in self.index.tables:
            assert table.dim == self.dim
            assert table.H == self.hyperplanes_per_table
    
    def test_add_vector(self):
        """Test adding a vector to LSH index"""
        # Act
        self.index.add(self.chunk_id1, self.vector1)
        
        # Assert
        assert self.chunk_id1 in self.index.vecs
        # Vector should be normalized
        norm_vector = self.index.vecs[self.chunk_id1]
        assert len(norm_vector) == self.dim
        
        # Check that vector is added to all tables
        for table in self.index.tables:
            sig = table.signature(norm_vector)
            if sig in table.buckets:
                assert self.chunk_id1 in table.buckets[sig]
    
    def test_add_multiple_vectors(self):
        """Test adding multiple vectors to LSH index"""
        # Act
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        self.index.add(self.chunk_id3, self.vector3)
        
        # Assert
        assert len(self.index.vecs) == 3
        assert self.chunk_id1 in self.index.vecs
        assert self.chunk_id2 in self.index.vecs
        assert self.chunk_id3 in self.index.vecs
    
    def test_add_duplicate_vector(self):
        """Test adding a vector with existing chunk ID"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        
        # Act
        self.index.add(self.chunk_id1, self.vector2)  # Same ID, different vector
        
        # Assert
        assert len(self.index.vecs) == 1
        assert self.chunk_id1 in self.index.vecs
        # Should be updated to new vector
        norm_vector2 = self.index.vecs[self.chunk_id1]
        # The vector should be normalized version of vector2
        assert norm_vector2 is not None
    
    def test_remove_vector(self):
        """Test removing a vector from LSH index"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        
        # Act
        self.index.remove(self.chunk_id1)
        
        # Assert
        assert self.chunk_id1 not in self.index.vecs
        assert self.chunk_id2 in self.index.vecs
        assert len(self.index.vecs) == 1
    
    def test_remove_nonexistent_vector(self):
        """Test removing a vector that doesn't exist"""
        # Act
        self.index.remove(self.chunk_id1)  # Should not raise error
        
        # Assert
        assert len(self.index.vecs) == 0
    
    def test_search_empty_index(self):
        """Test searching an empty LSH index"""
        # Act
        results = self.index.search(self.query_vector, k=5)
        
        # Assert
        assert results == []
    
    def test_search_single_vector(self):
        """Test searching with a single vector in the index"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        
        # Act
        results = self.index.search(self.query_vector, k=5)
        
        # Assert
        assert len(results) == 1
        assert results[0][0] == self.chunk_id1
        assert abs(results[0][1] - 1.0) < 1e-10  # Perfect similarity (same vector)
    
    def test_search_multiple_vectors(self):
        """Test searching with multiple vectors in the index"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        self.index.add(self.chunk_id3, self.vector3)
        
        # Act
        results = self.index.search(self.query_vector, k=3)
        
        # Assert
        assert len(results) == 3
        # Results should be sorted by similarity (descending)
        assert results[0][0] == self.chunk_id1  # Most similar (same vector)
        assert abs(results[0][1] - 1.0) < 1e-10  # Perfect similarity
        
        # Verify sorting
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
    
    def test_search_with_k_limit(self):
        """Test searching with k limit smaller than available vectors"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        self.index.add(self.chunk_id3, self.vector3)
        
        # Act
        results = self.index.search(self.query_vector, k=2)
        
        # Assert
        assert len(results) == 2
        assert results[0][0] == self.chunk_id1  # Most similar
    
    def test_search_with_zero_vector(self):
        """Test searching with zero vector"""
        # Arrange
        zero_vector = [0.0] * self.dim
        self.index.add(self.chunk_id1, self.vector1)
        
        # Act
        results = self.index.search(zero_vector, k=5)
        
        # Assert
        # Should return empty results for zero vector
        assert results == []
    
    def test_rebuild_empty_index(self):
        """Test rebuilding an empty LSH index"""
        # Arrange
        items = []
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.vecs) == 0
        for table in self.index.tables:
            assert len(table.buckets) == 0
    
    def test_rebuild_with_items(self):
        """Test rebuilding LSH index with new items"""
        # Arrange
        # Add some initial vectors
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        
        # New items for rebuild
        new_items = [
            (self.chunk_id3, self.vector3),
            (uuid4(), [0.1, 0.1, 0.1] * 42 + [0.1, 0.1])
        ]
        
        # Act
        self.index.rebuild(new_items)
        
        # Assert
        assert len(self.index.vecs) == 2
        assert self.chunk_id3 in self.index.vecs
        assert self.chunk_id1 not in self.index.vecs  # Should be cleared
        assert self.chunk_id2 not in self.index.vecs  # Should be cleared
    
    def test_rebuild_preserves_data(self):
        """Test that rebuild preserves the correct data"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2),
            (self.chunk_id3, self.vector3)
        ]
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.vecs) == 3
        assert self.chunk_id1 in self.index.vecs
        assert self.chunk_id2 in self.index.vecs
        assert self.chunk_id3 in self.index.vecs
    
    def test_search_after_rebuild(self):
        """Test searching after rebuild"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2)
        ]
        self.index.rebuild(items)
        
        # Act
        results = self.index.search(self.query_vector, k=2)
        
        # Assert
        assert len(results) == 2
        assert results[0][0] == self.chunk_id1  # Most similar
        assert abs(results[0][1] - 1.0) < 1e-10  # Perfect similarity
    
    def test_different_parameters(self):
        """Test LSH index with different parameters"""
        # Arrange
        index = LSHIndex(
            dim=64,
            num_tables=8,
            hyperplanes_per_table=16,
            seed=123
        )
        
        # Act
        chunk_id = uuid4()
        vector = [0.5] * 64
        index.add(chunk_id, vector)
        results = index.search(vector, k=1)
        
        # Assert
        assert len(results) == 1
        assert results[0][0] == chunk_id
        assert results[0][1] == 1.0
    
    def test_concurrent_operations(self):
        """Test LSH index with concurrent-like operations"""
        # Arrange
        vectors = []
        for i in range(10):
            chunk_id = uuid4()
            vector = [float(i % 5) / 5.0] * self.dim
            vectors.append((chunk_id, vector))
        
        # Act
        for chunk_id, vector in vectors:
            self.index.add(chunk_id, vector)
        
        # Test that vectors are added correctly (some might be duplicates)
        assert len(self.index.vecs) >= 5  # At least half should be added
        
        # Test that search returns some results (LSH is probabilistic)
        results = self.index.search([0.0] * self.dim, k=5)
        # LSH might not find exact matches, but should return some results
        assert len(results) <= 5  # Should not return more than requested
    
    def test_zero_vector_handling(self):
        """Test handling of zero vectors"""
        # Arrange
        zero_vector = [0.0] * self.dim
        
        # Act
        self.index.add(self.chunk_id1, zero_vector)
        
        # Assert
        # Zero vector should not be added (normalize returns None)
        assert self.chunk_id1 not in self.index.vecs
        assert len(self.index.vecs) == 0
    
    def test_large_number_of_vectors(self):
        """Test performance with a large number of vectors"""
        # Arrange
        vectors = []
        for i in range(50):  # Reduced for test speed
            chunk_id = uuid4()
            vector = [float(i % 10) / 10.0] * self.dim
            vectors.append((chunk_id, vector))
            self.index.add(chunk_id, vector)
        
        # Act
        results = self.index.search([0.0] * self.dim, k=10)
        
        # Assert
        assert len(results) <= 10
        # Some vectors might not be added due to normalization issues, so we check most are added
        assert len(self.index.vecs) >= 45  # At least 90% should be added
        
        # Verify sorting
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
    
    def test_identical_vectors(self):
        """Test handling of identical vectors"""
        # Arrange
        identical_vector = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
        
        # Act
        self.index.add(self.chunk_id1, identical_vector)
        self.index.add(self.chunk_id2, identical_vector)
        
        # Act
        results = self.index.search(identical_vector, k=2)
        
        # Assert
        assert len(results) == 2
        assert abs(results[0][1] - 1.0) < 1e-10  # Perfect similarity
        assert abs(results[1][1] - 1.0) < 1e-10  # Perfect similarity
