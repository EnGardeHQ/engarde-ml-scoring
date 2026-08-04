"""
Behavioral Data Collection API Router

Endpoints for collecting behavioral profiles used as ML training data.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from app.core.auth import verify_service_token, require_tenant
from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/behavioral", tags=["Behavioral Data"])


class CollectProfileRequest(BaseModel):
    tenant_id: str
    user_id: str


class CollectProfileResponse(BaseModel):
    success: bool
    profile: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/collect", response_model=CollectProfileResponse)
async def collect_behavioral_profile(
    request: CollectProfileRequest,
    _token: str = Depends(verify_service_token),
    db=Depends(get_db),
):
    """Collect and build a behavioral profile for ML training."""
    try:
        from app.services.behavioral_data_collector import BehavioralDataCollector

        collector = BehavioralDataCollector(db)
        profile = collector.collect_profile(
            tenant_id=request.tenant_id,
            user_id=request.user_id,
        )
        return CollectProfileResponse(
            success=True,
            profile=profile.__dict__ if hasattr(profile, '__dict__') else {"raw": str(profile)},
        )
    except Exception as e:
        logger.error("Profile collection failed: %s", e, exc_info=True)
        return CollectProfileResponse(success=False, error=str(e))
