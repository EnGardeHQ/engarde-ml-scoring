import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    env: str = os.getenv("ENV", "development")
    log_level: str = os.getenv("LOG_LEVEL", "info")
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    service_token: str = os.getenv("ML_SCORING_SERVICE_TOKEN", "")
    port: str = os.getenv("PORT", "8000")
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    bigquery_project_id: str = os.getenv("BIGQUERY_PROJECT_ID", "")
    bigquery_dataset_id: str = os.getenv("BIGQUERY_DATASET_ID", "engarde_analytics")
    bigquery_credentials_json: str = os.getenv("BIGQUERY_CREDENTIALS_JSON", "")
    ga4_service_account_json: str = os.getenv("GA4_SERVICE_ACCOUNT_JSON", "")
    token_wallet_url: str = os.getenv("TOKEN_WALLET_URL", "http://token-wallet.railway.internal:8080")
    token_wallet_service_token: str = os.getenv("TOKEN_WALLET_SERVICE_TOKEN", "")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)

    def validate_production(self) -> None:
        if self.env == "production":
            missing = [n for n, v in (
                ("ML_SCORING_SERVICE_TOKEN", self.service_token),
                ("DATABASE_URL", self.database_url),
                ("PORT", self.port),
            ) if not v]
            if missing:
                import logging
                logging.getLogger(__name__).warning(
                    "engarde-ml-scoring: missing env vars (degraded): %s", missing)

settings = Settings()
