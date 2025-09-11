from typing import List, Optional
from uuid import UUID
from datetime import datetime

from vector_db_api.models.indexing import IndexType
from vector_db_api.models.entities import Library
from vector_db_api.models.metadata import LibraryMetadata
from vector_db_api.repos.libraries import LibraryRepo
from vector_db_api.repos.documents import DocumentRepo
from vector_db_api.repos.chunks import ChunkRepo
from vector_db_api.concurrency.locks import LibraryLockRegistry
from vector_db_api.indexing.registry import IndexRegistry

from vector_db_api.services.exceptions import NotFoundError, ConflictError, ValidationError

class LibraryService:
    def __init__(self, libs: LibraryRepo, docs: DocumentRepo, chunks: ChunkRepo, locks: LibraryLockRegistry, indexes: IndexRegistry) -> None:
        self.libs = libs
        self.docs = docs
        self.chunks = chunks

        self.locks = locks
        self.indexes = indexes

    def create(self, name: str, embedding_dim: int, index_config: Optional[IndexType] = None, metadata = None) -> Library:
        # Handle metadata conversion similar to DocumentService
        if metadata is None:
            library_metadata = LibraryMetadata()
        elif isinstance(metadata, dict):
            library_metadata = LibraryMetadata(**metadata)
        else:
            library_metadata = metadata
            
        new_library = Library(
            name=name,
            embedding_dim=embedding_dim,
            index_config=index_config or IndexType(),
            metadata=library_metadata
        )

        self.libs.add(new_library)
        
        # Initialize empty index handle
        self.indexes.get_or_create(new_library)

        return new_library
    
    def get(self, lib_id: UUID) -> Library:
        lib = self.libs.get(lib_id)
        if not lib:
            raise NotFoundError(f"Library with id {lib_id} not found.")
        
        return lib
    
    def list(self) -> List[Library]:
        return self.libs.list()

    def update_config(self, lib_id: UUID, new_config: IndexType) -> Library:
        # Swap index type/config; rebuild from current chunks under write lock.
        lib = self.libs.get(lib_id)
        if not lib:
            raise NotFoundError(f"Library with id {lib_id} not found.")
        
        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_write()

        try:
            expected_version = lib.version
            lib.index_config = new_config
            lib.updated_at = datetime.utcnow()

            if not self.libs.update_on_version(lib, expected_version):
                raise ConflictError(f"Library with id {lib_id} has been modified concurrently.")
            
            # Rebuild index from chunks with embeddings
            new_index = self.indexes.create_index(lib.index_config, lib.embedding_dim)
            items = []
            offset = 0

            while True:
                batch = self.chunks.list_by_library(lib_id, limit=1000, offset=offset)
                if not batch:
                    break
                for chunk in batch:
                    if chunk.embedding is not None:
                        items.append((chunk.id, chunk.embedding))
                offset += 1000
            
            new_index.rebuild(items)
            self.indexes.swap(lib_id, new_index)
            return lib
        finally:
            lock.release_write()
    
    def delete(self, lib_id: UUID) -> None:
        # Delete library + documents & chunks + drop index handle, under write lock.
        lib = self.libs.get(lib_id)
        if not lib:
            return
        
        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_write()

        try:
            offset = 0
            while True:
                docs = self.docs.list_by_library(lib_id, limit=1000, offset=offset)
                if not docs:
                    break
                
                for doc in docs:
                    self.chunks.delete_by_document(doc.id)
                    self.docs.delete(doc.id)
                offset += 1000
            
            self.indexes.remove(lib_id)
            self.libs.delete(lib_id)
        finally:
            lock.release_write()
        