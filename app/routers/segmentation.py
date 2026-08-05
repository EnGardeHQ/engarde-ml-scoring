"""
Audience Segmentation API Router

Endpoints for ML-powered audience segmentation using clustering algorithms.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

from app.core.auth import verify_service_token, require_tenant, enforce_tenant_match

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/segmentation", tags=["Segmentation"])


class SegmentRequest(BaseModel):
    # [Story 16.4] Retained for caller compatibility but no longer trusted:
    # the X-EnGarde-Tenant-Id header is authoritative (403 on mismatch).
    tenant_id: Optional[str] = None
    data: List[Dict[str, Any]]
    algorithm: str = "kmeans"  # kmeans, dbscan, agglomerative, spectral, gaussian_mixture
    n_clusters: Optional[int] = 5
    features: Optional[List[str]] = None


class SegmentResponse(BaseModel):
    success: bool
    segments: Optional[List[Dict[str, Any]]] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/cluster", response_model=SegmentResponse)
async def run_segmentation(
    request: SegmentRequest,
    _token: str = Depends(verify_service_token),
    tenant_id: str = Depends(require_tenant),
):
    """Run clustering segmentation on provided data."""
    # [Story 16.4] Header-derived tenant wins over body-supplied tenant_id.
    tenant_id = enforce_tenant_match(tenant_id, request.tenant_id)
    try:
        from app.services.advanced_segmentation import AdvancedSegmentationEngine

        engine = AdvancedSegmentationEngine()
        result = engine.segment(
            data=request.data,
            algorithm=request.algorithm,
            n_clusters=request.n_clusters,
            features=request.features,
        )
        return SegmentResponse(
            success=True,
            segments=result.get("segments", []),
            metrics=result.get("metrics", {}),
        )
    except Exception as e:
        logger.error("Segmentation failed: %s", e, exc_info=True)
        return SegmentResponse(success=False, error=str(e))
