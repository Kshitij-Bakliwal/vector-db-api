from datetime import datetime
from fastapi import APIRouter
from vector_db_api.api.dto import HealthOut

router = APIRouter(tags=["health"])

@router.get("/health", response_model=HealthOut)
def health():
    return HealthOut(timestamp=datetime.utcnow(), details={})