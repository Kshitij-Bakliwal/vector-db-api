from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, ConfigDict

# ---------- Metadata (fixed but flexible via tags/custom) ----------
class BaseMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_uri: Optional[str] = None
    author: Optional[str] = None
    lang: Optional[str] = None       # e.g., 'en', 'en-US'
    mime_type: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class ChunkMetadata(BaseMetadata):
    page_number: Optional[int] = None
    token_count: Optional[int] = None
    sha256: Optional[str] = None

class DocumentMetadata(BaseMetadata):
    title: Optional[str] = None
    summary: Optional[str] = None
    sha256: Optional[str] = None

class LibraryMetadata(BaseMetadata):
    description: Optional[str] = None
