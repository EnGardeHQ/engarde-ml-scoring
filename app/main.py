from fastapi import FastAPI, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.auth import verify_service_token
from app.core.config import settings
from app.core.logging import configure_logging
from app.routers import health

configure_logging()
settings.validate_production()

app = FastAPI(
    title="EnGarde ML Scoring Service",
    version="0.1.0",
    docs_url="/docs" if settings.env != "production" else None,
    redoc_url="/redoc" if settings.env != "production" else None,
    openapi_url="/openapi.json" if settings.env != "production" else None,
)

app.include_router(health.router)

@app.get("/")
def root(_: str = Depends(verify_service_token)):
    return {"success": True, "data": {"service": "engarde-ml-scoring"}}

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"success": False, "detail": exc.detail})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(status_code=422, content={"success": False, "detail": exc.errors()})
