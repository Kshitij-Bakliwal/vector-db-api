"""
Unit tests for FlatIndex
"""

import pytest
from uuid import uuid4
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.indexing.flat import FlatIndex


class TestFlatIndex:
    """Test cases for FlatIndex"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.index = FlatIndex()
        self.chunk_id1 = uuid4()
        self.chunk_id2 = uuid4()
        self.chunk_id3 = uuid4()
        
        # Sample vectors (128 dimensions)
        self.vector1 = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
        self.vector2 = [0.4, 0.5, 0.6] * 42 + [0.4, 0.5]
        self.vector3 = [0.7, 0.8, 0.9] * 42 + [0.7, 0.8]
        self.query_vector = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]  # Similar to vector1
    
    def test_add_vector(self):
        """Test adding a vector to the index"""
        # Act
        self.index.add(self.chunk_id1, self.vector1)
        
        # Assert
        assert self.chunk_id1 in self.index.vectors
        assert self.index.vectors[self.chunk_id1] == self.vector1
    
    def test_add_multiple_vectors(self):
        """Test adding multiple vectors to the index"""
        # Act
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        self.index.add(self.chunk_id3, self.vector3)
        
        # Assert
        assert len(self.index.vectors) == 3
        assert self.index.vectors[self.chunk_id1] == self.vector1
        assert self.index.vectors[self.chunk_id2] == self.vector2
        assert self.index.vectors[self.chunk_id3] == self.vector3
    
    def test_update_existing_vector(self):
        """Test updating an existing vector"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        new_vector = [0.9, 0.8, 0.7] * 42 + [0.9, 0.8]
        
        # Act
        self.index.update(self.chunk_id1, new_vector)
        
        # Assert
        assert self.index.vectors[self.chunk_id1] == new_vector
        assert len(self.index.vectors) == 1
    
    def test_update_nonexistent_vector(self):
        """Test updating a vector that doesn't exist"""
        # Arrange
        new_vector = [0.9, 0.8, 0.7] * 42 + [0.9, 0.8]
        
        # Act
        self.index.update(self.chunk_id1, new_vector)
        
        # Assert
        assert self.index.vectors[self.chunk_id1] == new_vector
        assert len(self.index.vectors) == 1
    
    def test_remove_existing_vector(self):
        """Test removing an existing vector"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        
        # Act
        self.index.remove(self.chunk_id1)
        
        # Assert
        assert self.chunk_id1 not in self.index.vectors
        assert self.chunk_id2 in self.index.vectors
        assert len(self.index.vectors) == 1
    
    def test_remove_nonexistent_vector(self):
        """Test removing a vector that doesn't exist"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        
        # Act
        self.index.remove(self.chunk_id2)  # Doesn't exist
        
        # Assert
        assert self.chunk_id1 in self.index.vectors
        assert self.chunk_id2 not in self.index.vectors
        assert len(self.index.vectors) == 1
    
    def test_search_empty_index(self):
        """Test searching an empty index"""
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
        assert results[1][0] in [self.chunk_id2, self.chunk_id3]  # Second most similar
    
    def test_search_with_k_larger_than_vectors(self):
        """Test searching with k larger than available vectors"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        
        # Act
        results = self.index.search(self.query_vector, k=5)
        
        # Assert
        assert len(results) == 2  # Only 2 vectors available
        assert results[0][0] == self.chunk_id1
        assert results[1][0] == self.chunk_id2
    
    def test_search_similarity_scores(self):
        """Test that search returns correct similarity scores"""
        # Arrange
        # Create vectors with known similarities
        identical_vector = [1.0, 0.0, 0.0] * 42 + [1.0, 0.0]
        orthogonal_vector = [0.0, 1.0, 0.0] * 42 + [0.0, 1.0]
        opposite_vector = [-1.0, 0.0, 0.0] * 42 + [-1.0, 0.0]
        
        self.index.add(self.chunk_id1, identical_vector)
        self.index.add(self.chunk_id2, orthogonal_vector)
        self.index.add(self.chunk_id3, opposite_vector)
        
        query = [1.0, 0.0, 0.0] * 42 + [1.0, 0.0]
        
        # Act
        results = self.index.search(query, k=3)
        
        # Assert
        assert len(results) == 3
        # Should be sorted by similarity
        assert abs(results[0][1] - 1.0) < 1e-10   # Identical vector
        assert abs(results[1][1] - 0.0) < 1e-10   # Orthogonal vector
        assert abs(results[2][1] - (-1.0)) < 1e-10  # Opposite vector
    
    def test_rebuild_empty_index(self):
        """Test rebuilding an empty index"""
        # Arrange
        items = []
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.vectors) == 0
    
    def test_rebuild_with_items(self):
        """Test rebuilding index with new items"""
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
        assert len(self.index.vectors) == 2
        assert self.chunk_id3 in self.index.vectors
        assert self.chunk_id1 not in self.index.vectors  # Should be cleared
        assert self.chunk_id2 not in self.index.vectors  # Should be cleared
    
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
        assert len(self.index.vectors) == 3
        assert self.index.vectors[self.chunk_id1] == self.vector1
        assert self.index.vectors[self.chunk_id2] == self.vector2
        assert self.index.vectors[self.chunk_id3] == self.vector3
    
    def test_add_after_rebuild(self):
        """Test adding vectors after rebuild"""
        # Arrange
        items = [(self.chunk_id1, self.vector1)]
        self.index.rebuild(items)
        
        # Act
        self.index.add(self.chunk_id2, self.vector2)
        
        # Assert
        assert len(self.index.vectors) == 2
        assert self.index.vectors[self.chunk_id1] == self.vector1
        assert self.index.vectors[self.chunk_id2] == self.vector2
    
    def test_update_after_rebuild(self):
        """Test updating vectors after rebuild"""
        # Arrange
        items = [(self.chunk_id1, self.vector1)]
        self.index.rebuild(items)
        
        # Act
        self.index.update(self.chunk_id1, self.vector2)
        
        # Assert
        assert len(self.index.vectors) == 1
        assert self.index.vectors[self.chunk_id1] == self.vector2
    
    def test_remove_after_rebuild(self):
        """Test removing vectors after rebuild"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2)
        ]
        self.index.rebuild(items)
        
        # Act
        self.index.remove(self.chunk_id1)
        
        # Assert
        assert len(self.index.vectors) == 1
        assert self.chunk_id1 not in self.index.vectors
        assert self.chunk_id2 in self.index.vectors
    
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
    
    def test_duplicate_chunk_ids(self):
        """Test handling of duplicate chunk IDs"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        
        # Act
        self.index.add(self.chunk_id1, self.vector2)  # Same ID, different vector
        
        # Assert
        assert len(self.index.vectors) == 1
        assert self.index.vectors[self.chunk_id1] == self.vector2  # Should be updated
    
    def test_zero_vector_handling(self):
        """Test handling of zero vectors"""
        # Arrange
        zero_vector = [0.0] * 128
        
        # Act
        self.index.add(self.chunk_id1, zero_vector)
        results = self.index.search(zero_vector, k=1)
        
        # Assert
        assert len(results) == 1
        assert results[0][0] == self.chunk_id1
        assert results[0][1] == 0.0  # Zero vector similarity is 0 (handled by cosine_similarity)
    
    def test_different_vector_dimensions(self):
        """Test handling of vectors with different dimensions"""
        # Arrange
        short_vector = [1.0, 2.0, 3.0]
        long_vector = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Act
        self.index.add(self.chunk_id1, short_vector)
        self.index.add(self.chunk_id2, long_vector)
        
        # Assert
        assert len(self.index.vectors) == 2
        assert self.index.vectors[self.chunk_id1] == short_vector
        assert self.index.vectors[self.chunk_id2] == long_vector
    
    def test_large_number_of_vectors(self):
        """Test performance with a large number of vectors"""
        # Arrange
        vectors = []
        for i in range(100):
            chunk_id = uuid4()
            vector = [float(i % 10) / 10.0] * 128
            vectors.append((chunk_id, vector))
            self.index.add(chunk_id, vector)
        
        # Act
        results = self.index.search([0.0] * 128, k=10)
        
        # Assert
        assert len(results) == 10
        assert len(self.index.vectors) == 100
        
        # Verify sorting
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
