"""
Unit tests for IVFIndex
"""

import pytest
from uuid import uuid4
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.indexing.ivf import IVFIndex


class TestIVFIndex:
    """Test cases for IVFIndex"""
    
    def setup_method(self):
        """Set up test fixtures before each test method"""
        self.dim = 128
        self.num_centroids = 4
        self.nprobe = 2
        self.index = IVFIndex(
            dim=self.dim,
            num_centroids=self.num_centroids,
            nprobe=self.nprobe,
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
    
    def test_ivf_index_initialization(self):
        """Test IVF index initialization"""
        # Assert
        assert self.index.dim == self.dim
        assert self.index.k == self.num_centroids
        assert self.index.nprobe == self.nprobe
        assert len(self.index.centroids) == 0  # No centroids initially
        assert len(self.index.lists) == 0
        assert len(self.index.vecs) == 0
        assert len(self.index.assign) == 0
    
    def test_add_vector_without_centroids(self):
        """Test adding a vector when no centroids exist"""
        # Act
        self.index.add(self.chunk_id1, self.vector1)
        
        # Assert
        assert self.chunk_id1 in self.index.vecs
        assert self.chunk_id1 not in self.index.assign  # No assignment without centroids
        assert len(self.index.centroids) == 0
    
    def test_add_multiple_vectors_without_centroids(self):
        """Test adding multiple vectors when no centroids exist"""
        # Act
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        self.index.add(self.chunk_id3, self.vector3)
        
        # Assert
        assert len(self.index.vecs) == 3
        assert self.chunk_id1 in self.index.vecs
        assert self.chunk_id2 in self.index.vecs
        assert self.chunk_id3 in self.index.vecs
        assert len(self.index.assign) == 0  # No assignments without centroids
    
    def test_update_vector_without_centroids(self):
        """Test updating a vector when no centroids exist"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        
        # Act
        self.index.update(self.chunk_id1, self.vector2)
        
        # Assert
        assert self.chunk_id1 in self.index.vecs
        assert self.index.vecs[self.chunk_id1] != self.vector1  # Should be updated
        assert self.chunk_id1 not in self.index.assign
    
    def test_remove_vector_without_centroids(self):
        """Test removing a vector when no centroids exist"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        
        # Act
        self.index.remove(self.chunk_id1)
        
        # Assert
        assert self.chunk_id1 not in self.index.vecs
        assert self.chunk_id2 in self.index.vecs
        assert len(self.index.vecs) == 1
    
    def test_search_without_centroids(self):
        """Test searching when no centroids exist"""
        # Arrange
        self.index.add(self.chunk_id1, self.vector1)
        self.index.add(self.chunk_id2, self.vector2)
        
        # Act
        results = self.index.search(self.query_vector, k=2)
        
        # Assert
        assert len(results) == 2
        # Results should be sorted by similarity
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
    
    def test_rebuild_creates_centroids(self):
        """Test that rebuild creates centroids"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2),
            (self.chunk_id3, self.vector3)
        ]
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.centroids) > 0
        assert len(self.index.centroids) <= self.num_centroids
        assert len(self.index.vecs) == 3
        assert len(self.index.assign) == 3  # All vectors should be assigned
    
    def test_rebuild_with_single_vector(self):
        """Test rebuild with a single vector"""
        # Arrange
        items = [(self.chunk_id1, self.vector1)]
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.centroids) == 1  # Should create 1 centroid
        assert len(self.index.vecs) == 1
        assert len(self.index.assign) == 1
        assert self.chunk_id1 in self.index.assign
    
    def test_rebuild_with_empty_items(self):
        """Test rebuild with empty items"""
        # Arrange
        items = []
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.centroids) == 0
        assert len(self.index.vecs) == 0
        assert len(self.index.assign) == 0
        assert len(self.index.lists) == 0
    
    def test_add_after_rebuild(self):
        """Test adding vectors after rebuild"""
        # Arrange
        items = [(self.chunk_id1, self.vector1)]
        self.index.rebuild(items)
        
        # Act
        self.index.add(self.chunk_id2, self.vector2)
        
        # Assert
        assert len(self.index.vecs) == 2
        assert self.chunk_id2 in self.index.vecs
        assert self.chunk_id2 in self.index.assign  # Should be assigned to centroid
        assert len(self.index.assign) == 2
    
    def test_update_after_rebuild(self):
        """Test updating vectors after rebuild"""
        # Arrange
        items = [(self.chunk_id1, self.vector1)]
        self.index.rebuild(items)
        
        # Act
        self.index.update(self.chunk_id1, self.vector2)
        
        # Assert
        assert len(self.index.vecs) == 1
        assert self.chunk_id1 in self.index.vecs
        assert self.chunk_id1 in self.index.assign
    
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
        assert len(self.index.vecs) == 1
        assert self.chunk_id1 not in self.index.vecs
        assert self.chunk_id1 not in self.index.assign
        assert self.chunk_id2 in self.index.vecs
        assert self.chunk_id2 in self.index.assign
    
    def test_search_after_rebuild(self):
        """Test searching after rebuild"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2),
            (self.chunk_id3, self.vector3)
        ]
        self.index.rebuild(items)
        
        # Act
        results = self.index.search(self.query_vector, k=3)
        
        # Assert
        assert len(results) >= 2  # IVF might not find all vectors due to nprobe limitation
        # Results should be sorted by similarity
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
    
    def test_search_with_k_limit(self):
        """Test searching with k limit"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2),
            (self.chunk_id3, self.vector3)
        ]
        self.index.rebuild(items)
        
        # Act
        results = self.index.search(self.query_vector, k=2)
        
        # Assert
        assert len(results) == 2
        assert results[0][0] == self.chunk_id1  # Most similar
    
    def test_search_with_zero_vector(self):
        """Test searching with zero vector"""
        # Arrange
        zero_vector = [0.0] * self.dim
        items = [(self.chunk_id1, self.vector1)]
        self.index.rebuild(items)
        
        # Act
        results = self.index.search(zero_vector, k=5)
        
        # Assert
        # Should return empty results for zero vector
        assert results == []
    
    def test_centroid_assignment(self):
        """Test that vectors are assigned to correct centroids"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2),
            (self.chunk_id3, self.vector3)
        ]
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.assign) == 3
        for chunk_id in [self.chunk_id1, self.chunk_id2, self.chunk_id3]:
            assert chunk_id in self.index.assign
            centroid_id = self.index.assign[chunk_id]
            assert 0 <= centroid_id < len(self.index.centroids)
            assert chunk_id in self.index.lists[centroid_id]
    
    def test_centroid_lists_consistency(self):
        """Test that centroid lists are consistent with assignments"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2),
            (self.chunk_id3, self.vector3)
        ]
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        # Check that all assigned vectors are in their centroid lists
        for chunk_id, centroid_id in self.index.assign.items():
            assert chunk_id in self.index.lists[centroid_id]
        
        # Check that all vectors in lists are assigned
        for centroid_id, chunk_ids in self.index.lists.items():
            for chunk_id in chunk_ids:
                assert self.index.assign[chunk_id] == centroid_id
    
    def test_different_parameters(self):
        """Test IVF index with different parameters"""
        # Arrange
        index = IVFIndex(
            dim=64,
            num_centroids=8,
            nprobe=4,
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
    
    def test_large_number_of_vectors(self):
        """Test performance with a large number of vectors"""
        # Arrange
        vectors = []
        for i in range(20):  # Reduced for test speed
            chunk_id = uuid4()
            vector = [float(i % 10) / 10.0] * self.dim
            vectors.append((chunk_id, vector))
        
        # Act
        self.index.rebuild(vectors)
        results = self.index.search([0.0] * self.dim, k=10)
        
        # Assert
        assert len(results) <= 10
        # Some vectors might not be added due to normalization issues
        assert len(self.index.vecs) >= 18  # At least 90% should be added
        assert len(self.index.centroids) > 0
        
        # Verify sorting
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]
    
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
    
    def test_rebuild_with_zero_vectors(self):
        """Test rebuild with zero vectors in items"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, [0.0] * self.dim),  # Zero vector
            (self.chunk_id3, self.vector3)
        ]
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.vecs) == 2  # Only non-zero vectors
        assert self.chunk_id1 in self.index.vecs
        assert self.chunk_id2 not in self.index.vecs
        assert self.chunk_id3 in self.index.vecs
    
    def test_identical_vectors(self):
        """Test handling of identical vectors"""
        # Arrange
        identical_vector = [0.1, 0.2, 0.3] * 42 + [0.1, 0.2]
        items = [
            (self.chunk_id1, identical_vector),
            (self.chunk_id2, identical_vector)
        ]
        
        # Act
        self.index.rebuild(items)
        results = self.index.search(identical_vector, k=2)
        
        # Assert
        assert len(results) == 2
        assert abs(results[0][1] - 1.0) < 1e-10  # Perfect similarity
        assert abs(results[1][1] - 1.0) < 1e-10  # Perfect similarity
    
    def test_nprobe_limitation(self):
        """Test that nprobe limits the number of centroids searched"""
        # Arrange
        # Create vectors that would be assigned to different centroids
        vectors = []
        for i in range(10):
            chunk_id = uuid4()
            # Create vectors in different regions of space
            vector = [float(i % 5) / 5.0] * self.dim
            vectors.append((chunk_id, vector))
        
        self.index.rebuild(vectors)
        
        # Act
        results = self.index.search([0.0] * self.dim, k=5)
        
        # Assert
        # Should still return results, but limited by nprobe
        assert len(results) <= 5
        assert len(self.index.centroids) > 0
    
    def test_centroid_initialization(self):
        """Test that centroids are properly initialized"""
        # Arrange
        items = [
            (self.chunk_id1, self.vector1),
            (self.chunk_id2, self.vector2),
            (self.chunk_id3, self.vector3)
        ]
        
        # Act
        self.index.rebuild(items)
        
        # Assert
        assert len(self.index.centroids) > 0
        assert len(self.index.centroids) <= self.num_centroids
        
        # Check that centroids are normalized
        for centroid in self.index.centroids:
            assert len(centroid) == self.dim
            # Centroids should be normalized (or close to it)
            norm_sq = sum(x * x for x in centroid)
            assert abs(norm_sq - 1.0) < 1e-6  # Should be close to 1.0
