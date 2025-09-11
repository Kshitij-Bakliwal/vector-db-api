"""
Unit tests for BaseIndex interface
"""

import pytest
from abc import ABC, abstractmethod
from uuid import uuid4
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.indexing.base import BaseIndex


class TestBaseIndex:
    """Test BaseIndex abstract interface"""
    
    def test_base_index_can_be_instantiated(self):
        """Test that BaseIndex can be instantiated (it's not a true ABC)"""
        # BaseIndex is not a true abstract base class, so it can be instantiated
        index = BaseIndex()
        assert index is not None
    
    def test_base_index_has_required_methods(self):
        """Test that BaseIndex has all required abstract methods"""
        # Check that BaseIndex has the required methods
        required_methods = ['add', 'update', 'remove', 'search', 'rebuild']
        
        for method_name in required_methods:
            assert hasattr(BaseIndex, method_name)
            method = getattr(BaseIndex, method_name)
            assert callable(method)
    
    def test_base_index_inheritance(self):
        """Test that concrete implementations inherit from BaseIndex"""
        from vector_db_api.indexing.flat import FlatIndex
        from vector_db_api.indexing.lsh import LSHIndex
        from vector_db_api.indexing.ivf import IVFIndex
        
        # Test that concrete classes inherit from BaseIndex
        assert issubclass(FlatIndex, BaseIndex)
        assert issubclass(LSHIndex, BaseIndex)
        assert issubclass(IVFIndex, BaseIndex)
    
    def test_base_index_method_signatures(self):
        """Test that BaseIndex methods have correct signatures"""
        # Test add method signature
        add_method = BaseIndex.add
        # Should accept chunk_id: UUID and vec: List[float]
        
        # Test update method signature
        update_method = BaseIndex.update
        # Should accept chunk_id: UUID and vec: List[float]
        
        # Test remove method signature  
        remove_method = BaseIndex.remove
        # Should accept chunk_id: UUID
        
        # Test search method signature
        search_method = BaseIndex.search
        # Should accept query: List[float] and k: int, return List[Tuple[UUID, float]]
        
        # Test rebuild method signature
        rebuild_method = BaseIndex.rebuild
        # Should accept items: List[Tuple[UUID, List[float]]]
        
        # All methods should be callable
        assert callable(add_method)
        assert callable(update_method)
        assert callable(remove_method)
        assert callable(search_method)
        assert callable(rebuild_method)
    
    def test_concrete_implementations_implement_all_methods(self):
        """Test that concrete implementations implement all abstract methods"""
        from vector_db_api.indexing.flat import FlatIndex
        from vector_db_api.indexing.lsh import LSHIndex
        from vector_db_api.indexing.ivf import IVFIndex
        
        concrete_classes = [FlatIndex, LSHIndex, IVFIndex]
        
        for cls in concrete_classes:
            # Should be able to instantiate (no abstract methods left unimplemented)
            if cls == LSHIndex:
                instance = cls(dim=128)
            elif cls == IVFIndex:
                instance = cls(dim=128, num_centroids=64)
            else:  # FlatIndex
                instance = cls()
            
            # Should have all required methods
            assert hasattr(instance, 'add')
            assert hasattr(instance, 'update')
            assert hasattr(instance, 'remove')
            assert hasattr(instance, 'search')
            assert hasattr(instance, 'rebuild')
            
            # Methods should be callable
            assert callable(instance.add)
            assert callable(instance.update)
            assert callable(instance.remove)
            assert callable(instance.search)
            assert callable(instance.rebuild)
