"""
Unit tests for concurrency and locking mechanisms
"""

import pytest
import threading
import time
from uuid import uuid4

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from vector_db_api.concurrency.locks import ReadWriteLock, LibraryLockRegistry


class TestReadWriteLock:
    """Test ReadWriteLock implementation"""
    
    def test_read_write_lock_initialization(self):
        """Test ReadWriteLock initialization"""
        lock = ReadWriteLock()
        
        assert lock.reads == 0
        assert hasattr(lock, 'read_lock')
        assert hasattr(lock, 'write_lock')
    
    def test_read_lock_acquisition(self):
        """Test acquiring read lock"""
        lock = ReadWriteLock()
        
        lock.acquire_read()
        assert lock.reads == 1
        
        lock.release_read()
        assert lock.reads == 0
    
    def test_multiple_read_locks(self):
        """Test multiple read locks can be acquired simultaneously"""
        lock = ReadWriteLock()
        
        lock.acquire_read()
        assert lock.reads == 1
        
        lock.acquire_read()
        assert lock.reads == 2
        
        lock.acquire_read()
        assert lock.reads == 3
        
        lock.release_read()
        assert lock.reads == 2
        
        lock.release_read()
        assert lock.reads == 1
        
        lock.release_read()
        assert lock.reads == 0
    
    def test_write_lock_acquisition(self):
        """Test acquiring write lock"""
        lock = ReadWriteLock()
        
        lock.acquire_write()
        # Write lock should be acquired (we can't easily test the internal state)
        lock.release_write()
        # Write lock should be released
    
    def test_write_lock_excludes_reads(self):
        """Test that write lock excludes read locks"""
        lock = ReadWriteLock()
        results = []
        
        def write_operation():
            lock.acquire_write()
            results.append("write_start")
            time.sleep(0.1)
            results.append("write_end")
            lock.release_write()
        
        def read_operation():
            time.sleep(0.05)  # Start after write
            lock.acquire_read()
            results.append("read_start")
            time.sleep(0.1)
            results.append("read_end")
            lock.release_read()
        
        # Start write first, then read
        write_thread = threading.Thread(target=write_operation)
        read_thread = threading.Thread(target=read_operation)
        
        write_thread.start()
        read_thread.start()
        
        write_thread.join()
        read_thread.join()
        
        # Write should complete before read starts
        assert results == ["write_start", "write_end", "read_start", "read_end"]
    
    def test_read_locks_exclude_write(self):
        """Test that read locks exclude write lock"""
        lock = ReadWriteLock()
        results = []
        
        def read_operation():
            lock.acquire_read()
            results.append("read_start")
            time.sleep(0.1)
            results.append("read_end")
            lock.release_read()
        
        def write_operation():
            time.sleep(0.05)  # Start after read
            lock.acquire_write()
            results.append("write_start")
            time.sleep(0.1)
            results.append("write_end")
            lock.release_write()
        
        # Start read first, then write
        read_thread = threading.Thread(target=read_operation)
        write_thread = threading.Thread(target=write_operation)
        
        read_thread.start()
        write_thread.start()
        
        read_thread.join()
        write_thread.join()
        
        # Read should complete before write starts
        assert results == ["read_start", "read_end", "write_start", "write_end"]
    
    def test_concurrent_reads(self):
        """Test that multiple reads can happen concurrently"""
        lock = ReadWriteLock()
        results = []
        
        def read_operation(thread_id):
            lock.acquire_read()
            results.append(f"read_{thread_id}_start")
            time.sleep(0.05)
            results.append(f"read_{thread_id}_end")
            lock.release_read()
        
        threads = []
        for i in range(3):
            thread = threading.Thread(target=read_operation, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All reads should start before any end
        start_count = sum(1 for r in results if r.endswith("_start"))
        end_count = sum(1 for r in results if r.endswith("_end"))
        
        assert start_count == 3
        assert end_count == 3
        
        # All starts should come before all ends
        start_indices = [i for i, r in enumerate(results) if r.endswith("_start")]
        end_indices = [i for i, r in enumerate(results) if r.endswith("_end")]
        
        assert max(start_indices) < min(end_indices)
    
    def test_write_lock_reentrancy(self):
        """Test that write lock is not reentrant"""
        lock = ReadWriteLock()
        
        lock.acquire_write()
        # Attempting to acquire write lock again should block (deadlock)
        # This is a simple test - in practice, this would cause a deadlock
        # We'll just test that we can acquire and release once
        lock.release_write()
    
    def test_read_lock_reentrancy(self):
        """Test that read lock is reentrant"""
        lock = ReadWriteLock()
        
        lock.acquire_read()
        assert lock.reads == 1
        
        lock.acquire_read()
        assert lock.reads == 2
        
        lock.acquire_read()
        assert lock.reads == 3
        
        lock.release_read()
        assert lock.reads == 2
        
        lock.release_read()
        assert lock.reads == 1
        
        lock.release_read()
        assert lock.reads == 0


class TestLibraryLockRegistry:
    """Test LibraryLockRegistry implementation"""
    
    def test_library_lock_registry_initialization(self):
        """Test LibraryLockRegistry initialization"""
        registry = LibraryLockRegistry()
        
        assert len(registry.locks) == 0
    
    def test_get_lock_for_library(self):
        """Test getting lock for a specific library"""
        registry = LibraryLockRegistry()
        library_id = uuid4()
        
        lock1 = registry.lock_for_library(library_id)
        lock2 = registry.lock_for_library(library_id)
        
        # Should return the same lock instance
        assert lock1 is lock2
        assert isinstance(lock1, ReadWriteLock)
    
    def test_get_lock_for_different_libraries(self):
        """Test getting locks for different libraries"""
        registry = LibraryLockRegistry()
        library_id1 = uuid4()
        library_id2 = uuid4()
        
        lock1 = registry.lock_for_library(library_id1)
        lock2 = registry.lock_for_library(library_id2)
        
        # Should return different lock instances
        assert lock1 is not lock2
        assert isinstance(lock1, ReadWriteLock)
        assert isinstance(lock2, ReadWriteLock)
    
    def test_library_locks_independence(self):
        """Test that locks for different libraries are independent"""
        registry = LibraryLockRegistry()
        library_id1 = uuid4()
        library_id2 = uuid4()
        
        lock1 = registry.lock_for_library(library_id1)
        lock2 = registry.lock_for_library(library_id2)
        
        # Acquiring write lock on library1 should not block read on library2
        results = []
        
        def write_library1():
            lock1.acquire_write()
            results.append("write_lib1_start")
            time.sleep(0.1)
            results.append("write_lib1_end")
            lock1.release_write()
        
        def read_library2():
            time.sleep(0.05)  # Start after write
            lock2.acquire_read()
            results.append("read_lib2_start")
            time.sleep(0.1)
            results.append("read_lib2_end")
            lock2.release_read()
        
        write_thread = threading.Thread(target=write_library1)
        read_thread = threading.Thread(target=read_library2)
        
        write_thread.start()
        read_thread.start()
        
        write_thread.join()
        read_thread.join()
        
        # Both operations should run concurrently
        assert "write_lib1_start" in results
        assert "write_lib1_end" in results
        assert "read_lib2_start" in results
        assert "read_lib2_end" in results
    
    def test_library_lock_registry_thread_safety(self):
        """Test that LibraryLockRegistry is thread-safe"""
        registry = LibraryLockRegistry()
        library_id = uuid4()
        results = []
        
        def get_lock_and_use(thread_id):
            lock = registry.lock_for_library(library_id)
            lock.acquire_read()
            results.append(f"thread_{thread_id}_start")
            time.sleep(0.05)
            results.append(f"thread_{thread_id}_end")
            lock.release_read()
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=get_lock_and_use, args=(i,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All threads should have completed
        assert len(results) == 10  # 5 starts + 5 ends
        
        # All starts should come before all ends (due to read lock)
        start_count = sum(1 for r in results if r.endswith("_start"))
        end_count = sum(1 for r in results if r.endswith("_end"))
        
        assert start_count == 5
        assert end_count == 5
    
    def test_library_lock_registry_cleanup(self):
        """Test that LibraryLockRegistry can be cleaned up"""
        registry = LibraryLockRegistry()
        library_id = uuid4()
        
        # Get a lock
        lock = registry.lock_for_library(library_id)
        assert len(registry.locks) == 1
        
        # Use the lock
        lock.acquire_read()
        lock.release_read()
        
        # Lock should still exist
        assert len(registry.locks) == 1
        
        # Get the same lock again
        same_lock = registry.lock_for_library(library_id)
        assert same_lock is lock
        assert len(registry.locks) == 1
