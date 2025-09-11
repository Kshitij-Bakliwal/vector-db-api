from typing import List, Optional
from uuid import UUID
from datetime import datetime

from vector_db_api.models.entities import Document, Chunk
from vector_db_api.repos.libraries import LibraryRepo
from vector_db_api.repos.documents import DocumentRepo
from vector_db_api.repos.chunks import ChunkRepo
from vector_db_api.concurrency.locks import LibraryLockRegistry
from vector_db_api.indexing.registry import IndexRegistry
from vector_db_api.models.metadata import DocumentMetadata

from vector_db_api.services.exceptions import NotFoundError, ConflictError, ValidationError

class DocumentService:
    def __init__(self, libs: LibraryRepo, docs: DocumentRepo, chunks: ChunkRepo, locks: LibraryLockRegistry, indexes: IndexRegistry) -> None:
        self.libs = libs
        self.docs = docs
        self.chunks = chunks

        self.locks = locks
        self.indexes = indexes

    def create(self, lib_id: UUID, metadata = None) -> Document:
        lib = self.libs.get(lib_id)
        if not lib:
            raise NotFoundError(f"Library with id {lib_id} not found.")
        
        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_write()

        try:
            # Convert dict metadata to DocumentMetadata object
            if metadata is None:
                doc_metadata = DocumentMetadata()
            elif isinstance(metadata, dict):
                doc_metadata = DocumentMetadata(**metadata)
            else:
                doc_metadata = metadata
                
            new_document = Document(
                library_id=lib_id,
                metadata=doc_metadata
            )
            
            self.docs.add(new_document)
            return new_document
        finally:
            lock.release_write()
    
    def create_with_chunks(self, lib_id: UUID, chunks: List[Chunk], metadata = None) -> Document:
        lib = self.libs.get(lib_id)
        if not lib:
            raise NotFoundError(f"Library with id {lib_id} not found.")
        
        embedding_dim = lib.embedding_dim
        index = self.indexes.get_or_create(lib)

        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_write()

        try:
            # Convert dict metadata to DocumentMetadata object
            if metadata is None:
                doc_metadata = DocumentMetadata()
            elif isinstance(metadata, dict):
                doc_metadata = DocumentMetadata(**metadata)
            else:
                doc_metadata = metadata
                
            new_document = Document(
                library_id=lib_id,
                metadata=doc_metadata
            )

            self.docs.add(new_document)

            created_ids = []
            for chunk in chunks:
                chunk.library_id = lib_id
                chunk.document_id = new_document.id

                if chunk.embedding is not None and len(chunk.embedding) != embedding_dim:
                    raise ValidationError(f"Embedding dim mismatch: got {len(chunk.embedding)}, expected {embedding_dim}.")
                
                chunk.created_at = datetime.utcnow()
                chunk.updated_at = datetime.utcnow()

                self.chunks.add(chunk)
                created_ids.append(chunk.id)
                
                if chunk.embedding is not None:
                    index.add(chunk.id, chunk.embedding)
                
            new_document.chunk_ids.extend(created_ids)
            new_document.updated_at = datetime.utcnow()

            if not self.docs.update_on_version(new_document, 1):
                raise ConflictError(f"Document with id {new_document.id} modified concurrently during creation.")
            
            return new_document
        finally:
            lock.release_write()
    
    def get(self, lib_id: UUID, doc_id: UUID) -> Document:
        doc = self.docs.get(doc_id)
        if not doc or doc.library_id != lib_id:
            raise NotFoundError(f"Document with id {doc_id} not found.")
        
        return doc
    
    def list(self, lib_id: UUID, limit: int = 100, offset: int = 0, 
             has_tag: Optional[str] = None, created_after: Optional[datetime] = None,
             sort_by: str = "updated_at", order: str = "desc") -> List[Document]:
        return self.docs.list_by_library(lib_id, limit, offset, has_tag, created_after, sort_by, order)

    def update_metadata(self, lib_id: UUID, doc_id: UUID, new_metadata: DocumentMetadata) -> Document:
        doc = self.docs.get(doc_id)
        if not doc or doc.library_id != lib_id:
            raise NotFoundError(f"Document with id {doc_id} not found.")
        
        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_write()
        
        try:
            expected_version = doc.version
            
            for key, value in new_metadata.model_dump().items():
                setattr(doc.metadata, key, value)
            
            doc.updated_at = datetime.utcnow()

            if not self.docs.update_on_version(doc, expected_version):
                raise ConflictError(f"Document with id {doc_id} modified concurrently during update.")
            
            return doc
        finally:
            lock.release_write()
    
    def delete(self, lib_id: UUID, doc_id: UUID) -> None:
        doc = self.docs.get(doc_id)
        if not doc or doc.library_id != lib_id:
            return
        
        lock = self.locks.lock_for_library(lib_id)
        lock.acquire_write()

        try:
            # remove chunks in document and corresponding index entries
            index = self.indexes.get(lib_id)
            for chunk in self.chunks.list_by_document(doc_id, limit=10**9, offset=0):
                if index is not None and chunk.embedding is not None:
                    index.remove(chunk.id)
            
            self.chunks.delete_by_document(doc_id)
            self.docs.delete(doc_id)
        finally:
            lock.release_write()
    
    def move_to_library(self, doc_id: UUID, src_lib_id: UUID, dst_lib_id: UUID) -> Document:
        # Move a doc and all its chunks to a different library
        if src_lib_id == dst_lib_id:
            raise ValidationError(f"Source and destination libraries are the same.")
        
        src_lib = self.libs.get(src_lib_id)
        dst_lib = self.libs.get(dst_lib_id)
        if not src_lib or not dst_lib:
            raise NotFoundError(f"Source or destination library not found.")
        
        l1, l2 = sorted([src_lib_id, dst_lib_id], key=lambda x: str(x))
        l1_lock = self.locks.lock_for_library(l1)
        l2_lock = self.locks.lock_for_library(l2)

        l1_lock.acquire_write()
        l2_lock.acquire_write()

        try:
            doc = self.docs.get(doc_id)
            if not doc or doc.library_id != src_lib_id:
                raise NotFoundError(f"Document with id {doc_id} not found in source library.")
            
            expected_version = doc.version

            src_index = self.indexes.get(src_lib_id)
            dst_index = self.indexes.get_or_create(dst_lib)

            for chunk in self.chunks.list_by_document(doc_id, limit=10**9, offset=0):

                if src_index is not None and chunk.embedding is not None:
                    src_index.remove(chunk.id)
                
                chunk.library_id = dst_lib_id
                chunk.updated_at = datetime.utcnow()

                if not self.chunks.update_on_version(chunk, chunk.version):
                    raise ConflictError(f"Chunk with id {chunk.id} modified concurrently during move.")
                
                if chunk.embedding is not None:
                    if len(chunk.embedding) != dst_lib.embedding_dim:
                        raise ValidationError(f"Embedding dim mismatch for destination library.")
                    dst_index.add(chunk.id, chunk.embedding)
            
            doc.library_id = dst_lib_id
            doc.updated_at = datetime.utcnow()

            if not self.docs.update_on_version(doc, expected_version):
                raise ConflictError(f"Document with id {doc_id} modified concurrently during move.")
            
            return doc
        finally:
            l1_lock.release_write()
            l2_lock.release_write()
                
                
        
                