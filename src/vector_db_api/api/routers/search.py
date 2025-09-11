from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends

from vector_db_api.api.dto import (
    SearchIn,
    SearchOut,
    SearchHit
)

from vector_db_api.api.deps import get_search_svc
from vector_db_api.services.search import SearchService

router = APIRouter(prefix="/libraries/{lib_id}/search", tags=["search"])

# ========== routes ==========

@router.post("", response_model=SearchOut)
def search(
    lib_id: UUID,
    body: SearchIn,
    svc: SearchService = Depends(get_search_svc)
):
    hits = svc.query(
        lib_id=lib_id,
        query_embedding=body.query_embedding,
        k=body.k,
        metric=body.metric,
        filters=body.filters.model_dump() if body.filters else None
    )

    return SearchOut(results=[
        SearchHit(
            chunk_id=UUID(str(hit["chunk_id"])),
            document_id=UUID(str(hit["document_id"])),
            score=float(hit["score"]),
            text=hit["text"],
            metadata=hit["metadata"],
            created_at=hit["created_at"],
            updated_at=hit["updated_at"]
        ) for hit in hits]
    )