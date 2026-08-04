"""
Synthetic Enrichment Inference Engine
======================================
Real-time inference engine for predicting user attributes from behavioral signals.

This module loads trained ML models and performs real-time predictions on
user behavioral profiles to generate synthetic enrichment data.

CRITICAL: Models are loaded lazily to avoid slow app startup.
First prediction will be slower due to model loading.

Features predicted:
- Income tier (classification)
- Purchase propensity (regression)
- Interests (multi-label classification - TODO)
- Cultural affinity (multi-label classification - TODO)
"""

import numpy as np
from typing import Dict, Optional, List
import logging
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

# Global model cache
_MODEL_CACHE = {}
_MODEL_VERSION = None
_SKLEARN_IMPORTED = False


def _import_sklearn():
    """Lazy import scikit-learn and joblib"""
    global _SKLEARN_IMPORTED, joblib

    if not _SKLEARN_IMPORTED:
        logger.info("Loading scikit-learn for ML inference...")
        import joblib as jl
        joblib = jl
        _SKLEARN_IMPORTED = True
        logger.info("scikit-learn loaded successfully")


class SyntheticAttributes:
    """Predicted attributes from synthetic enrichment"""

    def __init__(
        self,
        income_tier: str,
        income_confidence: float,
        interests: Optional[List[Dict[str, float]]] = None,
        cultural_affinity: Optional[Dict[str, float]] = None,
        purchase_propensity: float = 0.0,
        purchase_confidence: float = 0.0,
        device_sophistication: str = "mid-tier",
        device_confidence: float = 0.5,
        life_stage: str = "established",
        life_stage_confidence: float = 0.5
    ):
        self.income_tier = income_tier
        self.income_confidence = income_confidence
        self.interests = interests or []
        self.cultural_affinity = cultural_affinity or {}
        self.purchase_propensity = purchase_propensity
        self.purchase_confidence = purchase_confidence
        self.device_sophistication = device_sophistication
        self.device_confidence = device_confidence
        self.life_stage = life_stage
        self.life_stage_confidence = life_stage_confidence

    def to_dict(self) -> Dict:
        """Convert to dictionary for API response"""
        return {
            "income_tier": self.income_tier,
            "income_confidence": self.income_confidence,
            "interests": self.interests,
            "cultural_affinity": self.cultural_affinity,
            "purchase_propensity": self.purchase_propensity,
            "purchase_confidence": self.purchase_confidence,
            "device_sophistication": self.device_sophistication,
            "device_confidence": self.device_confidence,
            "life_stage": self.life_stage,
            "life_stage_confidence": self.life_stage_confidence
        }


class SyntheticInferenceEngine:
    """
    Loads trained models and performs real-time predictions

    Models are loaded lazily from disk and cached in memory.
    Supports hot-swapping to new model versions.
    """

    def __init__(self, model_version: str = "latest", models_dir: str = "./models"):
        """
        Initialize inference engine

        Args:
            model_version: Model version to load ("latest" or specific version)
            models_dir: Base directory for model storage
        """
        self.model_version = model_version
        self.models_dir = models_dir
        self.models = {}
        self.metadata = {}
        self.data_sources = ["engarde_1p"]

        # Ensure sklearn is imported
        _import_sklearn()

    def _get_model_path(self) -> str:
        """
        Get path to model directory

        Returns:
            Path to model directory

        Raises:
            FileNotFoundError: If model directory doesn't exist
        """
        if self.model_version == "latest":
            # Find latest model version
            if not os.path.exists(self.models_dir):
                raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

            versions = [
                d for d in os.listdir(self.models_dir)
                if d.startswith("synthetic_v") and os.path.isdir(os.path.join(self.models_dir, d))
            ]

            if not versions:
                raise FileNotFoundError(f"No models found in {self.models_dir}")

            # Sort by version (assuming format: synthetic_vYYYYMMDD)
            versions.sort(reverse=True)
            latest = versions[0]
            model_path = os.path.join(self.models_dir, latest)
            logger.info(f"Using latest model version: {latest}")
        else:
            model_path = os.path.join(self.models_dir, f"synthetic_v{self.model_version}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model version not found: {model_path}")

        return model_path

    def _load_models(self) -> None:
        """
        Load all trained models from disk

        Models are cached in memory for fast subsequent predictions.
        """
        global _MODEL_CACHE, _MODEL_VERSION

        # Check if models are already cached
        if _MODEL_VERSION == self.model_version and _MODEL_CACHE:
            self.models = _MODEL_CACHE
            logger.info(f"Using cached models version {self.model_version}")
            return

        logger.info(f"Loading models version {self.model_version}...")

        model_path = self._get_model_path()

        # Load metadata
        metadata_file = os.path.join(model_path, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)
                logger.info(f"Loaded metadata: {self.metadata.get('training_date')}")

        # Load income model
        income_model_file = os.path.join(model_path, "income_model.pkl")
        if os.path.exists(income_model_file):
            self.models["income"] = joblib.load(income_model_file)
            logger.info("Loaded income tier classifier")

        # Load purchase propensity model
        propensity_model_file = os.path.join(model_path, "purchase_propensity_model.pkl")
        if os.path.exists(propensity_model_file):
            self.models["purchase_propensity"] = joblib.load(propensity_model_file)
            logger.info("Loaded purchase propensity regressor")

        # TODO: Load interests and cultural affinity models

        # Check if AdGrid enhanced mode
        if self.metadata.get("mode") == "adgrid_enhanced":
            self.data_sources.append("adgrid_publishers")

        # Cache models
        _MODEL_CACHE = self.models
        _MODEL_VERSION = self.model_version

        logger.info(f"Models loaded successfully: {list(self.models.keys())}")

    def _extract_features(self, behavioral_profile: Dict) -> np.ndarray:
        """
        Convert behavioral profile to feature vector

        Args:
            behavioral_profile: Dict with campaign_engagement, conversation_patterns, conversion_history

        Returns:
            numpy array of features
        """
        campaign = behavioral_profile.get("campaign_engagement", {})
        conversation = behavioral_profile.get("conversation_patterns", {})
        conversion = behavioral_profile.get("conversion_history", {})

        # Feature vector (must match training feature order)
        features = [
            campaign.get("email_open_rate", 0.0),
            campaign.get("email_click_rate", 0.0),
            campaign.get("total_campaigns_received", 0),
            conversation.get("response_rate", 0.0),
            conversation.get("avg_sentiment", 0.5),
            conversation.get("purchase_intent_score", 0.0),
            conversion.get("purchase_history", 0),
            conversion.get("avg_order_value", 0.0),
            conversion.get("lifetime_value", 0.0),
            conversion.get("churn_probability", 0.0)
        ]

        return np.array([features])

    async def predict(self, behavioral_profile: Dict) -> SyntheticAttributes:
        """
        Run inference on all models

        Args:
            behavioral_profile: Dict with behavioral signals

        Returns:
            SyntheticAttributes with predictions and confidence scores
        """
        # Load models if not already loaded
        if not self.models:
            self._load_models()

        # Extract features
        features = self._extract_features(behavioral_profile)

        # Income tier prediction
        income_tier = "<50K"
        income_confidence = 0.0

        if "income" in self.models:
            try:
                income_pred = self.models["income"].predict(features)[0]
                income_prob = self.models["income"].predict_proba(features)[0]
                income_confidence = float(income_prob.max())
                income_tier = income_pred
            except Exception as e:
                logger.error(f"Error predicting income tier: {e}")

        # Purchase propensity prediction
        purchase_propensity = 0.0
        purchase_confidence = 0.0

        if "purchase_propensity" in self.models:
            try:
                purchase_prop = self.models["purchase_propensity"].predict(features)[0]
                purchase_propensity = float(max(0.0, min(1.0, purchase_prop)))  # Clip to [0, 1]
                purchase_confidence = 0.8  # TODO: Calculate actual confidence
            except Exception as e:
                logger.error(f"Error predicting purchase propensity: {e}")

        # Device sophistication (heuristic based on features)
        device_sophistication = self._infer_device_sophistication(behavioral_profile)
        device_confidence = 0.7

        # Life stage (heuristic based on features)
        life_stage = self._infer_life_stage(behavioral_profile)
        life_stage_confidence = 0.6

        logger.info(
            f"Inference complete: income={income_tier} ({income_confidence:.2f}), "
            f"propensity={purchase_propensity:.2f}"
        )

        return SyntheticAttributes(
            income_tier=income_tier,
            income_confidence=income_confidence,
            interests=[],  # TODO: Implement interests prediction
            cultural_affinity={},  # TODO: Implement cultural affinity prediction
            purchase_propensity=purchase_propensity,
            purchase_confidence=purchase_confidence,
            device_sophistication=device_sophistication,
            device_confidence=device_confidence,
            life_stage=life_stage,
            life_stage_confidence=life_stage_confidence
        )

    def _infer_device_sophistication(self, behavioral_profile: Dict) -> str:
        """
        Infer device sophistication from behavioral signals (heuristic)

        Args:
            behavioral_profile: Behavioral signals

        Returns:
            Device sophistication tier (budget, mid-tier, premium)
        """
        conversion = behavioral_profile.get("conversion_history", {})
        avg_order_value = conversion.get("avg_order_value", 0.0)

        # Simple heuristic based on spending
        if avg_order_value > 200:
            return "premium"
        elif avg_order_value > 75:
            return "mid-tier"
        else:
            return "budget"

    def _infer_life_stage(self, behavioral_profile: Dict) -> str:
        """
        Infer life stage from behavioral signals (heuristic)

        Args:
            behavioral_profile: Behavioral signals

        Returns:
            Life stage (early_career, established, executive)
        """
        conversion = behavioral_profile.get("conversion_history", {})
        lifetime_value = conversion.get("lifetime_value", 0.0)
        purchase_count = conversion.get("purchase_history", 0)

        # Simple heuristic based on engagement
        if lifetime_value > 1000 and purchase_count > 10:
            return "executive"
        elif lifetime_value > 300 or purchase_count > 3:
            return "established"
        else:
            return "early_career"

    def get_model_info(self) -> Dict:
        """
        Get information about loaded models

        Returns:
            Dict with model metadata
        """
        return {
            "version": self.model_version,
            "models_loaded": list(self.models.keys()),
            "data_sources": self.data_sources,
            "metadata": self.metadata
        }
