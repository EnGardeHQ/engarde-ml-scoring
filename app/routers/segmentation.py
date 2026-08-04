"""
Audience Segmentation API Router

Endpoints for ML-powered audience segmentation using clustering algorithms.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

from app.core.auth import verify_service_token, require_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/segmentation", tags=["Segmentation"])


class SegmentRequest(BaseModel):
    tenant_id: str
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
):
    """Run clustering segmentation on provided data."""
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
