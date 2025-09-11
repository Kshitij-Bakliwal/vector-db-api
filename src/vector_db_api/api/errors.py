from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from vector_db_api.services.exceptions import NotFoundError, ConflictError, ValidationError

def register_exception_handlers(app: FastAPI):
    @app.exception_handler(NotFoundError)
    async def _nf(_: Request, e: NotFoundError): return JSONResponse({"detail": str(e)}, status_code=404)
    @app.exception_handler(ConflictError)
    async def _cf(_: Request, e: ConflictError): return JSONResponse({"detail": str(e)}, status_code=409)
    @app.exception_handler(ValidationError)
    async def _ve(_: Request, e: ValidationError): return JSONResponse({"detail": str(e)}, status_code=422)