"""
ML Scoring API Router

Endpoints for running inference against trained models and triggering model training.
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging

from app.core.auth import verify_service_token, require_tenant, enforce_tenant_match
from app.ml.synthetic_inference import SyntheticInferenceEngine
from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/scoring", tags=["ML Scoring"])

inference_engine = SyntheticInferenceEngine()


class PredictRequest(BaseModel):
    # [Story 16.4] Retained for caller compatibility but no longer trusted:
    # the X-EnGarde-Tenant-Id header is authoritative (403 on mismatch).
    tenant_id: Optional[str] = None
    profile: Dict[str, Any]
    model_version: Optional[int] = None


class PredictResponse(BaseModel):
    success: bool
    predictions: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class TrainRequest(BaseModel):
    # [Story 16.4] Retained for caller compatibility but no longer trusted:
    # the X-EnGarde-Tenant-Id header is authoritative (403 on mismatch).
    tenant_id: Optional[str] = None
    model_types: Optional[List[str]] = None  # ["income", "purchase_propensity"]


class TrainResponse(BaseModel):
    success: bool
    message: str
    metrics: Optional[Dict[str, Any]] = None


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    _token: str = Depends(verify_service_token),
    tenant_id: str = Depends(require_tenant),
):
    """Run inference on a behavioral profile."""
    # [Story 16.4] Header-derived tenant wins over body-supplied tenant_id.
    tenant_id = enforce_tenant_match(tenant_id, request.tenant_id)
    try:
        result = inference_engine.predict(
            profile=request.profile,
            model_version=request.model_version,
        )
        return PredictResponse(success=True, predictions=result.__dict__ if hasattr(result, '__dict__') else {"raw": str(result)})
    except FileNotFoundError as e:
        logger.warning("Model not found: %s", e)
        return PredictResponse(success=False, error=f"Model not found: {e}")
    except Exception as e:
        logger.error("Prediction failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@router.post("/train", response_model=TrainResponse)
async def train_models(
    request: TrainRequest,
    _token: str = Depends(verify_service_token),
    tenant_id: str = Depends(require_tenant),
    db=Depends(get_db),
):
    """Trigger model training on behavioral profile data."""
    # [Story 16.4] Header-derived tenant wins over body-supplied tenant_id.
    tenant_id = enforce_tenant_match(tenant_id, request.tenant_id)
    try:
        from app.ml.synthetic_trainer import SyntheticEnrichmentTrainer
        trainer = SyntheticEnrichmentTrainer(db=db)
        results = await trainer.train_all()
        return TrainResponse(
            success=True,
            message="Training complete",
            metrics={k: str(v) for k, v in (results or {}).items()},
        )
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training failed: {e}")
