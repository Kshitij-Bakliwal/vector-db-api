from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vector_db_api.concurrency.locks import LibraryLockRegistry
from vector_db_api.indexing.registry import IndexRegistry

from vector_db_api.repos.libraries import LibraryRepo
from vector_db_api.repos.documents import DocumentRepo
from vector_db_api.repos.chunks import ChunkRepo

from vector_db_api.services.library import LibraryService
from vector_db_api.services.document import DocumentService
from vector_db_api.services.chunk import ChunkService
from vector_db_api.services.search import SearchService

from vector_db_api.api.routers import libraries as lib_router
from vector_db_api.api.routers import documents as doc_router
from vector_db_api.api.routers import chunks as chunk_router
from vector_db_api.api.routers import search as search_router
from vector_db_api.api.routers import health as health_router

from vector_db_api.api.errors import register_exception_handlers


def create_app() -> FastAPI:

    app = FastAPI(title="Vector DB REST API", version="1.0.0")

    libs = LibraryRepo()
    docs = DocumentRepo()
    chunks = ChunkRepo()

    locks = LibraryLockRegistry()
    indexes = IndexRegistry()

    app.state.container = {
        "library_svc": LibraryService(libs, docs, chunks, locks, indexes),
        "document_svc": DocumentService(libs, docs, chunks, locks, indexes),
        "chunk_svc": ChunkService(libs, docs, chunks, locks, indexes),
        "search_svc": SearchService(libs, chunks, locks, indexes)
    }

    app.include_router(lib_router.router)
    app.include_router(doc_router.router)
    app.include_router(chunk_router.router)
    app.include_router(search_router.router)
    app.include_router(health_router.router)

    register_exception_handlers(app)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _bootstrap_indexes() -> None:
        for lib in libs.list():
            index = indexes.get_or_create(lib)
            items = []
            offset = 0
            batch_size = 1000
            while True:
                batch = chunks.list_by_library(lib.id, limit=batch_size, offset=offset)
                if not batch:
                    break
                for chunk in batch:
                    if chunk.embedding is not None:
                        items.append((chunk.id, chunk.embedding))
                offset += batch_size
            index.rebuild(items)


    return app


app = create_app()