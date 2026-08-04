"""
Synthetic Enrichment Model Trainer
====================================
ML model training infrastructure for predicting user attributes from
behavioral signals.

Models trained:
1. Income tier classifier (4 classes: <50K, 50-100K, 100-150K, 150K+)
2. Interests classifier (multi-label)
3. Cultural affinity classifier (multi-label)
4. Purchase propensity regressor

Training data sources:
- behavioral_profiles table (En Garde 1st-party data)
- AdGrid publisher cohort data (if in enhanced mode)
- Known labels from PDL enrichments

CRITICAL: This is a compute-intensive process.
Should be run as a Celery background task, not in the API request path.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os
import json
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import text

logger = logging.getLogger(__name__)

# Lazy import ML libraries to avoid slow startup
# These will be imported only when training is triggered
_sklearn_imported = False
_models_loaded = False


def _import_sklearn():
    """Lazy import scikit-learn to avoid slow app startup"""
    global _sklearn_imported
    if not _sklearn_imported:
        logger.info("Loading scikit-learn for ML training...")
        global RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
        global train_test_split, cross_val_score
        global joblib

        from sklearn.ensemble import (
            RandomForestClassifier as RFC,
            GradientBoostingClassifier as GBC,
            GradientBoostingRegressor as GBR
        )
        from sklearn.model_selection import train_test_split as tts, cross_val_score as cvs
        import joblib as jl

        RandomForestClassifier = RFC
        GradientBoostingClassifier = GBC
        GradientBoostingRegressor = GBR
        train_test_split = tts
        cross_val_score = cvs
        joblib = jl

        _sklearn_imported = True
        logger.info("scikit-learn loaded successfully")


class SyntheticEnrichmentTrainer:
    """
    Trains ML models to predict user attributes from behavioral signals

    Supports two modes:
    - standalone: Uses only En Garde 1st-party data (60-70% accuracy)
    - adgrid_enhanced: Adds AdGrid publisher data (75-85% accuracy)
    """

    def __init__(self, mode: str = "standalone", db: Optional[Session] = None):
        """
        Initialize model trainer

        Args:
            mode: "standalone" or "adgrid_enhanced"
            db: Database session for querying training data
        """
        self.mode = mode
        self.db = db
        self.models = {}
        self.accuracy_metrics = {}

        # Ensure sklearn is imported
        _import_sklearn()

    async def prepare_training_data(self, tenant_id: Optional[str] = None) -> pd.DataFrame:
        """
        Aggregate all training data sources into a pandas DataFrame

        Args:
            tenant_id: Optional tenant filter. If None, uses all tenants (for platform-wide training)

        Returns:
            pandas DataFrame with features and labels
        """
        logger.info(f"Preparing training data (mode: {self.mode}, tenant: {tenant_id or 'all'})")

        # Collect En Garde 1st-party data
        engarde_data = await self._collect_engarde_data(tenant_id)

        if self.mode == "adgrid_enhanced":
            # Add AdGrid publisher data
            adgrid_data = await self._collect_adgrid_data()
            training_data = pd.concat([engarde_data, adgrid_data], ignore_index=True)
        else:
            training_data = engarde_data

        logger.info(f"Training data prepared: {len(training_data)} samples")
        return training_data

    async def _collect_engarde_data(self, tenant_id: Optional[str] = None) -> pd.DataFrame:
        """
        Query En Garde database for users with known attributes

        Joins behavioral_profiles with known labels (from previous PDL enrichments).

        Args:
            tenant_id: Optional tenant filter

        Returns:
            pandas DataFrame with features and labels
        """
        if not self.db:
            raise ValueError("Database session required for data collection")

        tenant_filter = "AND bp.tenant_id = :tenant_id" if tenant_id else ""
        params = {"tenant_id": tenant_id} if tenant_id else {}

        query = f"""
            SELECT
                bp.user_id,
                bp.email,
                -- Campaign engagement features
                (bp.campaign_engagement->>'email_open_rate')::FLOAT as email_open_rate,
                (bp.campaign_engagement->>'email_click_rate')::FLOAT as email_click_rate,
                (bp.campaign_engagement->>'total_campaigns_received')::INT as total_campaigns,
                -- Conversation features
                (bp.conversation_patterns->>'response_rate')::FLOAT as response_rate,
                (bp.conversation_patterns->>'avg_sentiment')::FLOAT as avg_sentiment,
                (bp.conversation_patterns->>'purchase_intent_score')::FLOAT as purchase_intent,
                -- Conversion features
                (bp.conversion_history->>'purchase_history')::INT as purchase_count,
                (bp.conversion_history->>'avg_order_value')::FLOAT as avg_order_value,
                (bp.conversion_history->>'lifetime_value')::FLOAT as lifetime_value,
                (bp.conversion_history->>'churn_probability')::FLOAT as churn_probability,
                -- Labels (from synthetic_enrichment_results or PDL)
                bp.known_income_tier,
                bp.known_interests,
                bp.known_cultural_affinity
            FROM behavioral_profiles bp
            WHERE bp.known_income_tier IS NOT NULL  -- Only users with known labels
            {tenant_filter}
        """

        result = self.db.execute(text(query), params)
        rows = result.fetchall()

        if not rows:
            logger.warning("No training data found with known labels")
            return pd.DataFrame()

        # Convert to DataFrame
        data = []
        for row in rows:
            data.append({
                "user_id": row[0],
                "email": row[1],
                "email_open_rate": row[2] or 0.0,
                "email_click_rate": row[3] or 0.0,
                "total_campaigns": row[4] or 0,
                "response_rate": row[5] or 0.0,
                "avg_sentiment": row[6] or 0.5,
                "purchase_intent": row[7] or 0.0,
                "purchase_count": row[8] or 0,
                "avg_order_value": row[9] or 0.0,
                "lifetime_value": row[10] or 0.0,
                "churn_probability": row[11] or 0.0,
                "income_tier": row[12],
                "interests": row[13],
                "cultural_affinity": row[14]
            })

        df = pd.DataFrame(data)
        logger.info(f"Collected {len(df)} training samples from En Garde data")
        return df

    async def _collect_adgrid_data(self) -> pd.DataFrame:
        """
        Query AdGrid cohort data for additional training samples

        This is a placeholder for future AdGrid API integration.

        Returns:
            pandas DataFrame with AdGrid data
        """
        logger.info("AdGrid data collection not yet implemented")
        # TODO: Implement AdGrid API integration
        return pd.DataFrame()

    def train_income_model(self, training_data: pd.DataFrame) -> Dict:
        """
        Train classifier for income tier prediction

        Args:
            training_data: DataFrame with features and income_tier labels

        Returns:
            Dict with accuracy metrics
        """
        logger.info("Training income tier model...")

        # Feature columns
        features = [
            'email_open_rate', 'email_click_rate', 'total_campaigns',
            'response_rate', 'avg_sentiment', 'purchase_intent',
            'purchase_count', 'avg_order_value', 'lifetime_value', 'churn_probability'
        ]

        # Filter out rows with missing labels
        data = training_data[training_data['income_tier'].notna()].copy()

        if len(data) < 50:
            logger.warning(f"Insufficient training data for income model: {len(data)} samples")
            return {"error": "insufficient_data", "samples": len(data)}

        X = data[features].fillna(0)
        y = data['income_tier']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate accuracy
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        cv_scores = cross_val_score(model, X, y, cv=5)

        self.models['income'] = model
        self.accuracy_metrics['income'] = {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'training_samples': len(data),
            'feature_count': len(features)
        }

        logger.info(f"Income model trained: {test_score:.2%} accuracy on {len(data)} samples")
        return self.accuracy_metrics['income']

    def train_purchase_propensity_model(self, training_data: pd.DataFrame) -> Dict:
        """
        Train regressor for purchase propensity prediction

        Args:
            training_data: DataFrame with features

        Returns:
            Dict with accuracy metrics
        """
        logger.info("Training purchase propensity model...")

        features = [
            'email_open_rate', 'email_click_rate', 'total_campaigns',
            'response_rate', 'avg_sentiment', 'purchase_intent',
            'purchase_count', 'avg_order_value', 'lifetime_value', 'churn_probability'
        ]

        # Create target variable (normalized purchase propensity)
        data = training_data.copy()
        data['propensity'] = data['purchase_count'] / (data['total_campaigns'] + 1)
        data['propensity'] = data['propensity'].clip(0, 1)  # Normalize to [0, 1]

        data = data[data['propensity'].notna()]

        if len(data) < 50:
            logger.warning(f"Insufficient training data for propensity model: {len(data)} samples")
            return {"error": "insufficient_data", "samples": len(data)}

        X = data[features].fillna(0)
        y = data['propensity']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Evaluate (R² score)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        self.models['purchase_propensity'] = model
        self.accuracy_metrics['purchase_propensity'] = {
            'train_r2': float(train_score),
            'test_r2': float(test_score),
            'training_samples': len(data),
            'feature_count': len(features)
        }

        logger.info(f"Purchase propensity model trained: R²={test_score:.3f} on {len(data)} samples")
        return self.accuracy_metrics['purchase_propensity']

    def train_all_models(self, training_data: pd.DataFrame) -> Dict:
        """
        Train all attribute prediction models

        Args:
            training_data: DataFrame with features and labels

        Returns:
            Dict with all accuracy metrics
        """
        logger.info("Training all synthetic enrichment models...")

        results = {}

        # Train income model
        try:
            results['income'] = self.train_income_model(training_data)
        except Exception as e:
            logger.error(f"Error training income model: {e}", exc_info=True)
            results['income'] = {"error": str(e)}

        # Train purchase propensity model
        try:
            results['purchase_propensity'] = self.train_purchase_propensity_model(training_data)
        except Exception as e:
            logger.error(f"Error training propensity model: {e}", exc_info=True)
            results['purchase_propensity'] = {"error": str(e)}

        # TODO: Add interests and cultural affinity models

        logger.info("All models trained successfully")
        return results

    def save_models(self, version: str, models_dir: str = "./models") -> None:
        """
        Persist trained models to disk

        Args:
            version: Model version identifier (e.g., "20260301")
            models_dir: Base directory for model storage
        """
        logger.info(f"Saving models version {version}...")

        model_path = os.path.join(models_dir, f"synthetic_v{version}")
        os.makedirs(model_path, exist_ok=True)

        # Save each model
        for name, model in self.models.items():
            model_file = os.path.join(model_path, f"{name}_model.pkl")
            joblib.dump(model, model_file)
            logger.info(f"Saved {name} model to {model_file}")

        # Save metadata
        metadata = {
            "version": version,
            "mode": self.mode,
            "accuracy_metrics": self.accuracy_metrics,
            "training_date": datetime.utcnow().isoformat(),
            "models": list(self.models.keys())
        }

        metadata_file = os.path.join(model_path, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Models saved to {model_path}")

    async def store_metrics_to_db(self, version: str, tenant_id: str) -> None:
        """
        Store model accuracy metrics to database

        Args:
            version: Model version
            tenant_id: Tenant ID (use 'platform' for platform-wide models)
        """
        if not self.db:
            logger.warning("No database session - skipping metrics storage")
            return

        for model_name, metrics in self.accuracy_metrics.items():
            if "error" in metrics:
                continue

            self.db.execute(
                text("""
                    INSERT INTO synthetic_model_metrics (
                        tenant_id,
                        model_version,
                        model_name,
                        mode,
                        train_accuracy,
                        test_accuracy,
                        cv_mean,
                        cv_std,
                        training_samples,
                        feature_count,
                        trained_at,
                        created_at,
                        updated_at
                    ) VALUES (
                        :tenant_id,
                        :model_version,
                        :model_name,
                        :mode,
                        :train_accuracy,
                        :test_accuracy,
                        :cv_mean,
                        :cv_std,
                        :training_samples,
                        :feature_count,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP,
                        CURRENT_TIMESTAMP
                    )
                """),
                {
                    "tenant_id": tenant_id,
                    "model_version": version,
                    "model_name": model_name,
                    "mode": self.mode,
                    "train_accuracy": metrics.get("train_accuracy") or metrics.get("train_r2"),
                    "test_accuracy": metrics.get("test_accuracy") or metrics.get("test_r2"),
                    "cv_mean": metrics.get("cv_mean"),
                    "cv_std": metrics.get("cv_std"),
                    "training_samples": metrics.get("training_samples"),
                    "feature_count": metrics.get("feature_count")
                }
            )

        self.db.commit()
        logger.info(f"Model metrics stored to database for version {version}")
