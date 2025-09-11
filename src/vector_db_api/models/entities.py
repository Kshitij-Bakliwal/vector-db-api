from typing import List, Optional, Dict
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict, field_validator

from vector_db_api.models.metadata import ChunkMetadata, DocumentMetadata, LibraryMetadata
from vector_db_api.models.indexing import IndexType

"""
Core Entity Models
"""
class Chunk(BaseModel):

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    id: UUID = Field(default_factory=uuid4)
    library_id: UUID
    document_id: UUID
    position: int = 0
    text: str = Field(..., min_length=1)
    embedding: Optional[List[float]] = None # validation at service layer
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1 # update on write


class Document(BaseModel):

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    id: UUID = Field(default_factory=uuid4)
    library_id: UUID
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    chunk_ids: List[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1


class Library(BaseModel):

    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    
    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1)
    embedding_dim: int = Field(..., gt=0)
    index_config: IndexType = Field(default_factory=IndexType)
    metadata: LibraryMetadata = Field(default_factory=LibraryMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1
    
    @field_validator("embedding_dim")
    @classmethod
    def _dim_reasonable(cls, v: int) -> int:
        # sanity check for dimension value
        if v > 8192:
            raise ValueError("embedding_dim too large")
        return v



