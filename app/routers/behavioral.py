"""
Behavioral Data Collection API Router

Endpoints for collecting behavioral profiles used as ML training data.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging

from app.core.auth import verify_service_token, require_tenant, enforce_tenant_match
from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/behavioral", tags=["Behavioral Data"])


class CollectProfileRequest(BaseModel):
    # [Story 16.4] Retained for caller compatibility but no longer trusted:
    # the X-EnGarde-Tenant-Id header is authoritative (403 on mismatch).
    tenant_id: Optional[str] = None
    user_id: str


class CollectProfileResponse(BaseModel):
    success: bool
    profile: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/collect", response_model=CollectProfileResponse)
async def collect_behavioral_profile(
    request: CollectProfileRequest,
    _token: str = Depends(verify_service_token),
    tenant_id: str = Depends(require_tenant),
    db=Depends(get_db),
):
    """Collect and build a behavioral profile for ML training."""
    # [Story 16.4] Header-derived tenant wins over body-supplied tenant_id.
    tenant_id = enforce_tenant_match(tenant_id, request.tenant_id)
    try:
        from app.services.behavioral_data_collector import BehavioralDataCollector

        collector = BehavioralDataCollector(db)
        profile = collector.collect_profile(
            tenant_id=tenant_id,
            user_id=request.user_id,
        )
        return CollectProfileResponse(
            success=True,
            profile=profile.__dict__ if hasattr(profile, '__dict__') else {"raw": str(profile)},
        )
    except Exception as e:
        logger.error("Profile collection failed: %s", e, exc_info=True)
        return CollectProfileResponse(success=False, error=str(e))
