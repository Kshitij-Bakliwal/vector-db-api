from __future__ import annotations

from typing import List, Optional
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, Query, status

from vector_db_api.api.dto import (
    DocumentCreate,
    DocumentCreateWithChunks,
    DocumentOut,
    DocumentListQuery,
    Page, PageMetadata,
    BulkChunksIn, BulkChunksOut,
    MoveDocumentIn,
    ChunkIn
)

from vector_db_api.api.deps import get_document_svc
from vector_db_api.services.document import DocumentService
from vector_db_api.models.entities import Chunk
from vector_db_api.models.metadata import ChunkMetadata

router = APIRouter(prefix="/libraries/{lib_id}/documents", tags=["documents"])

# ========== helpers ==========

def _get_document_response(doc: Document) -> DocumentOut:
    return DocumentOut(
        id=doc.id,
        library_id=doc.library_id,
        # external_id=doc.external_id,
        metadata=doc.metadata.model_dump() if hasattr(doc.metadata, 'model_dump') else doc.metadata,
        chunk_ids=doc.chunk_ids,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        version=doc.version
    )

def _chunk_request_to_entity(chunk: ChunkIn, lib_id: UUID, doc_id: UUID) -> Chunk:
    # Convert dict metadata to ChunkMetadata object
    if chunk.metadata is None:
        chunk_metadata = ChunkMetadata()
    elif isinstance(chunk.metadata, dict):
        chunk_metadata = ChunkMetadata(**chunk.metadata)
    else:
        chunk_metadata = chunk.metadata
    
    # Create chunk data, only include id if it's not None
    chunk_data = {
        "library_id": lib_id,
        "document_id": doc_id,
        "text": chunk.text,
        "position": chunk.position,
        "embedding": chunk.embedding,
        "metadata": chunk_metadata
    }
    
    # Only include id if it's provided (not None)
    if chunk.id is not None:
        chunk_data["id"] = chunk.id
        
    return Chunk(**chunk_data)

# ========== routes ==========

@router.post("", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def create_document(
    lib_id: UUID,
    body: DocumentCreate,
    svc: DocumentService = Depends(get_document_svc)
):
    doc = svc.create(
        lib_id=lib_id,
        # external_id=body.external_id,
        metadata=body.metadata)

    return _get_document_response(doc)


@router.post("/with-chunks", response_model=DocumentOut, status_code=status.HTTP_201_CREATED)
def create_document_with_chunks(
    lib_id: UUID,
    body: DocumentCreateWithChunks,
    svc: DocumentService = Depends(get_document_svc)
):
    tmp_doc = svc.create(
        lib_id=lib_id,
        # external_id=body.external_id,
        metadata=body.metadata)
    
    chunks = [_chunk_request_to_entity(chunk, lib_id, tmp_doc.id) for chunk in body.chunks]
    doc = svc.create_with_chunks(
        lib_id=lib_id,
        # external_id=body.external_id,
        chunks=chunks,
        metadata=body.metadata)

    return _get_document_response(doc)


@router.get("", response_model=Page[DocumentOut])
def list_documents(
    lib_id: UUID,
    limit: int = Query(50, gt=0, le=1000),
    offset: int = Query(0, ge=0),
    has_tag: Optional[str] = Query(None, description="Filter by tag"),
    created_after: Optional[datetime] = Query(None, description="Filter documents created after this date"),
    sort_by: Optional[str] = Query("updated_at", description="Sort by field"),
    order: Optional[str] = Query("desc", description="Sort order"),
    svc: DocumentService = Depends(get_document_svc)
):
    # Request one extra item to check if there are more pages
    docs = svc.list(lib_id, limit + 1, offset, has_tag, created_after, sort_by, order)
    has_more = len(docs) > limit
    
    # Remove the extra item if it exists
    if has_more:
        docs = docs[:limit]

    return Page[DocumentOut](
        items=[_get_document_response(doc) for doc in docs],
        page=PageMetadata(limit=limit, offset=offset, has_more=has_more)
    )
    

@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(
    lib_id: UUID,
    doc_id: UUID,
    svc: DocumentService = Depends(get_document_svc)
):
    doc = svc.get(lib_id, doc_id)
    return _get_document_response(doc)


@router.post("/{doc_id}:move", response_model=DocumentOut)
def move_document(
    lib_id: UUID,
    doc_id: UUID,
    body: MoveDocumentIn,
    svc: DocumentService = Depends(get_document_svc)
):
    doc = svc.move_to_library(doc_id, lib_id, body.dst_library_id)
    return _get_document_response(doc)


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    lib_id: UUID,
    doc_id: UUID,
    svc: DocumentService = Depends(get_document_svc)
):
    svc.delete(lib_id, doc_id)