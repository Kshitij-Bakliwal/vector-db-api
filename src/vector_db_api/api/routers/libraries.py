from __future__ import annotations

from typing import List
from uuid import UUID
from datetime import datetime

from fastapi import APIRouter, Depends, Query, status

from vector_db_api.api.dto import (
    LibraryCreate, LibraryOut, IndexConfigIn, IndexConfigOut,
    Page, PageMetadata, RebuildIndexOut
)

from vector_db_api.api.deps import get_library_svc
from vector_db_api.services.library import LibraryService
from vector_db_api.models.indexing import IndexType

router = APIRouter(prefix="/libraries", tags=["libraries"])

# ========== helpers ==========

def _get_library_response(lib: Library) -> LibraryOut:

    index_config = IndexConfigOut(
        type=lib.index_config.type,
        lsh=getattr(lib.index_config, "lsh", None),
        ivf=getattr(lib.index_config, "ivf", None))
    
    return LibraryOut(
        id=lib.id,
        name=lib.name,
        embedding_dim=lib.embedding_dim,
        index_config=index_config,
        metadata=getattr(lib.metadata, "model_dump", lambda: lib.metadata)(),
        created_at=lib.created_at,
        updated_at=lib.updated_at,
        version=lib.version
    )

# ========== routes ==========

@router.post("", response_model=LibraryOut, status_code=status.HTTP_201_CREATED)
def create_library(
    body: LibraryCreate,
    svc: LibraryService = Depends(get_library_svc)
):
    lib = svc.create(
        name=body.name,
        embedding_dim=body.embedding_dim,
        index_config=IndexType(
            type=body.index_config.type,
            lsh_num_tables=getattr(body.index_config.lsh, "num_tables", None) if body.index_config.lsh else None,
            lsh_hyperplanes_per_table=getattr(body.index_config.lsh, "hyperplanes_per_table", None) if body.index_config.lsh else None,
            ivf_num_centroids=getattr(body.index_config.ivf, "num_centroids", None) if body.index_config.ivf else None,
            ivf_nprobe=getattr(body.index_config.ivf, "nprobe", None) if body.index_config.ivf else None
        ),
        metadata=body.metadata
    )
    return _get_library_response(lib)


@router.get("", response_model=Page[LibraryOut])
def list_libraries(
    limit: int = Query(50, gt=0, le=1000),
    offset: int = Query(0, ge=0),
    svc: LibraryService = Depends(get_library_svc)
):
    all_libs = svc.list()
    # Request one extra item to check if there are more pages
    batch = all_libs[offset: offset + limit + 1]
    has_more = len(batch) > limit
    
    # Remove the extra item if it exists
    if has_more:
        batch = batch[:limit]
    
    items = [_get_library_response(lib) for lib in batch]

    return Page[LibraryOut](
        items=items, 
        page=PageMetadata(limit=limit, offset=offset, has_more=has_more))


@router.get("/{lib_id}", response_model=LibraryOut)
def get_library(
    lib_id: UUID,
    svc: LibraryService = Depends(get_library_svc)
):
    lib = svc.get(lib_id)
    return _get_library_response(lib)


@router.patch("/{lib_id}/index-config", response_model=LibraryOut)
def update_index_config(
    lib_id: UUID,
    body: IndexConfigIn,
    svc: LibraryService = Depends(get_library_svc)
):
    new_config = IndexType(
        type=body.type,
        lsh_num_tables=getattr(body.lsh, "num_tables", None) if body.lsh else None,
        lsh_hyperplanes_per_table=getattr(body.lsh, "hyperplanes_per_table", None) if body.lsh else None,
        ivf_num_centroids=getattr(body.ivf, "num_centroids", None) if body.ivf else None,
        ivf_nprobe=getattr(body.ivf, "nprobe", None) if body.ivf else None
    )
    lib = svc.update_config(lib_id, new_config)
    return _get_library_response(lib)


@router.post("/{lib_id}/rebuild-index", response_model=RebuildIndexOut)
def rebuild_index(
    lib_id: UUID,
    svc: LibraryService = Depends(get_library_svc)
):
    lib = svc.get(lib_id)
    lib = svc.update_config(lib_id, lib.index_config)

    return RebuildIndexOut(
        library_id=lib.id,
        index_type=lib.index_config.type,
        rebuild_at=datetime.utcnow()
    )


@router.delete("/{lib_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_library(
    lib_id: UUID,
    svc: LibraryService = Depends(get_library_svc)
):
    svc.delete(lib_id)


    
