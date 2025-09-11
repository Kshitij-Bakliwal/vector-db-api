from typing import Dict, List, Optional
from uuid import UUID
from datetime import datetime
from vector_db_api.models.entities import Document

class DocumentRepo:
    def __init__(self) -> None:
        self.documents: Dict[UUID, Document] = {}
        self.documents_by_library: Dict[UUID, List[UUID]] = {}
    
    def add(self, doc: Document) -> None:
        now = datetime.utcnow()
        doc.created_at = now
        doc.updated_at = now
        self.documents[doc.id] = doc
        self.documents_by_library.setdefault(doc.library_id, []).append(doc.id)
    
    def get(self, doc_id: UUID) -> Optional[Document]:
        doc = self.documents.get(doc_id)
        return doc.model_copy(deep=True) if doc else None
    
    def list_by_library(self, lib_id: UUID, limit: int = 100, offset: int = 0, 
                       has_tag: Optional[str] = None, created_after: Optional[datetime] = None, 
                       sort_by: str = "updated_at", order: str = "desc") -> List[Document]:
        docs = self.documents_by_library.get(lib_id, [])
        
        # Get all documents for this library
        all_docs = [self.documents[doc_id].model_copy(deep=True) for doc_id in docs]
        
        # Apply tag filtering if specified
        if has_tag is not None:
            all_docs = [doc for doc in all_docs if has_tag in doc.metadata.tags]
        
        # Apply date filtering if specified
        if created_after is not None:
            all_docs = [doc for doc in all_docs if doc.created_at > created_after]
        
        # Apply sorting
        reverse = (order == "desc")
        if sort_by == "created_at":
            all_docs.sort(key=lambda doc: doc.created_at, reverse=reverse)
        elif sort_by == "updated_at":
            all_docs.sort(key=lambda doc: doc.updated_at, reverse=reverse)
        
        # Apply pagination
        selected_docs = all_docs[offset: offset + limit]
        return selected_docs

    def update_on_version(self, doc: Document, expected_version: int) -> bool:
        current_doc = self.documents.get(doc.id)
        
        if current_doc is None or current_doc.version != expected_version:
            return False
        
        # remove from library if library has changed for document
        if current_doc.library_id != doc.library_id:
            old_docs = self.documents_by_library.get(current_doc.library_id, [])
            if doc.id in old_docs:
                old_docs.remove(doc.id)
            self.documents_by_library.setdefault(doc.library_id, []).append(doc.id)
        
        doc.version = expected_version + 1
        doc.updated_at = datetime.utcnow()
        self.documents[doc.id] = doc
        return True
    
    def delete(self, doc_id: UUID) -> bool:
        doc = self.documents.pop(doc_id, None)
        if not doc:
            return False
        docs_in_lib = self.documents_by_library.get(doc.library_id, [])
        if doc_id in docs_in_lib:
            docs_in_lib.remove(doc_id)
        return True
    