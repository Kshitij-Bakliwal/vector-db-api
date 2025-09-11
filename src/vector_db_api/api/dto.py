from __future__ import annotations

from typing import Any, List, Dict, Optional, Generic, TypeVar, Literal
from uuid import UUID
from datetime import datetime

from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

# ========== Shared ==========

T = TypeVar('T')

class APIError(BaseModel):
    code: str = Field(..., description="Error code")
    detail: str = Field(..., description="Error detail")
    meta: Optional[Dict[str, Any]] = Field(default=None)

class PageMetadata(BaseModel):
    limit: int = Field(..., gt=0, le=1000)
    offset: int = Field(..., ge=0)
    has_more: bool = Field(description="Whether more pages exist")

class Page(GenericModel, Generic[T]):
    items: List[T]
    page: PageMetadata

# ========== Indexing ==========

IndexType = Literal["flat", "lsh", "ivf"]
SimilarityMetric = Literal["cosine", "euclidean", "dot_product"]
SortOrder = Literal["asc", "desc"]

class LSHParams(BaseModel):
    num_tables: int = Field(8, ge=1, le=128)
    hyperplanes_per_table: int = Field(16, ge=1, le=512)

class IVFParams(BaseModel):
    num_centroids: int = Field(64, ge=2, le=65536)
    nprobe: int = Field(4, ge=1, le=1024)
    max_kmeans_iters: int = Field(2, ge=1, le=1000)

class IndexConfigIn(BaseModel):
    type: IndexType = "flat"
    lsh: Optional[LSHParams] = None
    ivf: Optional[IVFParams] = None

class IndexConfigOut(IndexConfigIn):
    pass

# ========== Chunk ==========

class ChunkIn(BaseModel):
    id: Optional[UUID] = None
    text: str = Field(..., min_length=1)
    position: int = Field(0, ge=0)
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = Field(default=None)

class ChunkUpdate(BaseModel):
    # partial update (PATCH)
    text: Optional[str] = Field(None, min_length=1)
    position: Optional[int] = Field(None, ge=0)
    embedding: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChunkOut(BaseModel):
    id: UUID
    library_id: UUID
    document_id: UUID
    text: str
    position: int
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int

# ========== Bulk Operations ==========

class BulkChunksIn(BaseModel):
    chunks: List[ChunkIn] = Field(default_factory=list, min_items=1)

class BulkChunksOut(BaseModel):
    chunk_ids: List[UUID]

# ========== Document ==========

class DocumentCreate(BaseModel):
    metadata: Optional[Dict[str, Any]] = Field(default=None)

class DocumentUpdate(BaseModel):
    # partial update (PATCH)

    metadata: Optional[Dict[str, Any]] = None

class DocumentCreateWithChunks(DocumentCreate):
    chunks: List[ChunkIn] = Field(default_factory=list, min_items=1)

class DocumentOut(BaseModel):
    id: UUID
    library_id: UUID
    # external_id: Optional[str] = None
    metadata: Dict[str, Any]
    chunk_ids: List[UUID]
    created_at: datetime
    updated_at: datetime
    version: int

class DocumentListQuery(BaseModel):
    # Query params for listing documents in a library
    limit: int = Field(50, ge=0, le=1000)
    offset: int = Field(0, ge=0)
    has_tag: Optional[str] = Field(default=None, description="Filter by tag")
    created_after: Optional[datetime] = Field(default=None, description="Filter documents created after this date")
    sort_by: Optional[Literal["created_at", "updated_at"]] = "updated_at"
    order: SortOrder = "desc"

# ========== Library ==========

class LibraryCreate(BaseModel):
    name: str = Field(..., min_length=1)
    embedding_dim: int = Field(..., gt=0, le=8192)
    index_config: IndexConfigIn = Field(default_factory=IndexConfigIn)
    metadata: Optional[Dict[str, Any]] = Field(default=None)

class LibraryUpdate(BaseModel):
    # partial update (PATCH)
    name: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict[str, Any]] = None
    index_config: Optional[IndexConfigIn] = None

class LibraryOut(BaseModel):
    id: UUID
    name: str
    embedding_dim: int
    index_config: IndexConfigOut
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    version: int

# ========== Search ==========

class SearchFilters(BaseModel):
    # Chunk-level post-filtering fields; extend as needed
    doc_ids: Optional[List[UUID]] = None
    tags: Optional[List[str]] = None
    author: Optional[str] = None
    created_after: Optional[datetime] = None

class SearchIn(BaseModel):
    query_embedding: List[float]
    k: int = Field(10, gt=0, le=1000)
    metric: SimilarityMetric = "cosine"
    filters: Optional[SearchFilters] = None

class SearchHit(BaseModel):
    chunk_id: UUID
    document_id: UUID
    score: float
    text: str
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class SearchOut(BaseModel):
    results: List[SearchHit]

# ========== Maintainence / Additional ==========

class RebuildIndexOut(BaseModel):
    library_id: UUID
    index_type: IndexType
    rebuild_at: datetime

class MoveDocumentIn(BaseModel):
    # document_id: UUID
    # src_library_id: UUID
    dst_library_id: UUID

class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    timestamp: datetime
    details: Dict[str, Any] = Field(default_factory=dict)