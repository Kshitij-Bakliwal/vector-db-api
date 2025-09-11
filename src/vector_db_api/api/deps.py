from __future__ import annotations

from typing import Any, Dict, cast
from fastapi import Depends, Request

from vector_db_api.services.library import LibraryService
from vector_db_api.services.document import DocumentService
from vector_db_api.services.chunk import ChunkService
from vector_db_api.services.search import SearchService

CONTAINER_KEY = "container"

def _get_container(request: Request) -> Dict[str, Any]:
    # Fetch the app-scoped container
    container = getattr(request.app.state, CONTAINER_KEY, None)
    if container is None:
        raise RuntimeError(
            "Service container not initialized."
            "Ensure app.state.container is set in src/main.create_app()"
        )

    return cast(Dict[str, Any], container)

def get_library_svc(container: Dict[str, Any] = Depends(_get_container)) -> LibraryService:
    svc = container.get("library_svc")
    if svc is None:
        raise RuntimeError("library_svc not found in container.")
    return cast(LibraryService, svc)

def get_document_svc(container: Dict[str, Any] = Depends(_get_container)) -> DocumentService:
    svc = container.get("document_svc")
    if svc is None:
        raise RuntimeError("document_svc not found in container.")
    return cast(DocumentService, svc)

def get_chunk_svc(container: Dict[str, Any] = Depends(_get_container)) -> ChunkService:
    svc = container.get("chunk_svc")
    if svc is None:
        raise RuntimeError("chunk_svc not found in container.")
    return cast(ChunkService, svc)

def get_search_svc(container: Dict[str, Any] = Depends(_get_container)) -> SearchService:
    svc = container.get("search_svc")
    if svc is None:
        raise RuntimeError("search_svc not found in container.")
    return cast(SearchService, svc)
