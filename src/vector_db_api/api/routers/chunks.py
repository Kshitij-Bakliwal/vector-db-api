from __future__ import annotations

from typing import List, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, Query, status

from vector_db_api.api.dto import (
    ChunkIn,
    ChunkOut,
    BulkChunksIn,
    BulkChunksOut
)

from vector_db_api.api.deps import get_chunk_svc
from vector_db_api.services.chunk import ChunkService
from vector_db_api.models.entities import Chunk
from vector_db_api.models.metadata import ChunkMetadata

router = APIRouter(prefix="/libraries/{lib_id}/documents/{doc_id}/chunks", tags=["chunks"])

# ========== helpers ==========

def _get_chunk_response(chunk: Chunk) -> ChunkOut:
    return ChunkOut(
        id=chunk.id,
        library_id=chunk.library_id,
        document_id=chunk.document_id,
        text=chunk.text,
        position=chunk.position,
        metadata=chunk.metadata.model_dump() if hasattr(chunk.metadata, 'model_dump') else chunk.metadata,
        created_at=chunk.created_at,
        updated_at=chunk.updated_at,
        version=chunk.version
    )

def _chunk_request_to_entity(payload: ChunkIn, lib_id: UUID, doc_id: UUID) -> Chunk:
    # Convert dict metadata to ChunkMetadata object
    if payload.metadata is None:
        chunk_metadata = ChunkMetadata()
    elif isinstance(payload.metadata, dict):
        chunk_metadata = ChunkMetadata(**payload.metadata)
    else:
        chunk_metadata = payload.metadata
    
    data: Dict[str, Any] = {
        "library_id": lib_id,
        "document_id": doc_id,
        "text": payload.text,
        "position": payload.position,
        "embedding": payload.embedding,
        "metadata": chunk_metadata
    }

    if payload.id is not None:
        data["id"] = payload.id

    return Chunk(**data)

# ========== routes ==========

@router.post("", response_model=ChunkOut)
def upsert_chunk(
    lib_id: UUID,
    doc_id: UUID,
    body: ChunkIn,
    svc: ChunkService = Depends(get_chunk_svc)
):
    """
    Create or update a single chunk (id optional).
    - Validates embedding dim against the library.
    - Updates index under the library write lock.
    """
    chunk = _chunk_request_to_entity(body, lib_id, doc_id)
    updated_chunk = svc.upsert(chunk)
    return _get_chunk_response(updated_chunk)


@router.post(":bulk", response_model=BulkChunksOut, status_code=status.HTTP_201_CREATED)
def bulk_upsert_chunks(
    lib_id: UUID,
    doc_id: UUID,
    body: BulkChunksIn,
    svc: ChunkService = Depends(get_chunk_svc)
):
    """
    Bulk create chunks (and index them). Each ChunkIn may omit `id`.
    Returns created/updated chunk ids.
    """
    chunks = [_chunk_request_to_entity(chunk, lib_id, doc_id) for chunk in body.chunks]
    updated_chunks = svc.bulk_upsert(lib_id, doc_id, chunks)
    return BulkChunksOut(chunk_ids=[chunk.id for chunk in updated_chunks])


@router.delete("/{chunk_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_chunk(
    lib_id: UUID,
    doc_id: UUID,
    chunk_id: UUID,
    svc: ChunkService = Depends(get_chunk_svc)
):
    svc.delete(lib_id, chunk_id)