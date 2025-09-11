import threading
from collections import defaultdict
from uuid import UUID

class ReadWriteLock:
    def __init__(self) -> None:
        self.reads = 0
        self.read_lock = threading.Lock()
        self.write_lock = threading.Lock()
    
    def acquire_read(self) -> None:
        with self.read_lock:
            self.reads += 1
            if self.reads == 1:
                self.write_lock.acquire()
    
    def release_read(self) -> None:
        with self.read_lock:
            self.reads -= 1
            if self.reads == 0:
                self.write_lock.release()
    
    def acquire_write(self) -> None:
        self.write_lock.acquire()
    
    def release_write(self) -> None:
        self.write_lock.release()

class LibraryLockRegistry:
    def __init__(self) -> None:
        self.locks = defaultdict(ReadWriteLock)
    
    def lock_for_library(self, lib_id: UUID) -> ReadWriteLock:
        return self.locks[lib_id]