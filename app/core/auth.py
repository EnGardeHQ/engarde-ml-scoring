import hmac

from fastapi import Header, HTTPException

from app.core.config import settings


async def verify_service_token(authorization: str = Header(default="")) -> str:
    """Service-to-service auth (subs-sync SUBS_SYNC_SERVICE_TOKEN pattern).

    The wallet service is private-network-only; producer services and the
    production-backend proxy authenticate with a shared bearer token.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    expected = settings.service_token
    if not expected or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid service token")
    return token


async def require_admin(x_engarde_role: str = Header(default="")) -> str:
    """The production-backend proxy resolves the caller's tenant JWT and
    forwards the verified role in X-EnGarde-Role. This service trusts the
    proxy (private network only) and never parses SSO JWTs itself.
    """
    if x_engarde_role not in ("admin", "superuser"):
        raise HTTPException(status_code=403, detail="Admin role required")
    return x_engarde_role


async def require_tenant(x_engarde_tenant_id: str = Header(default="")) -> str:
    if not x_engarde_tenant_id:
        raise HTTPException(status_code=400, detail="Missing X-EnGarde-Tenant-Id")
    return x_engarde_tenant_id


def enforce_tenant_match(header_tenant_id: str, claimed_tenant_id=None) -> str:
    """[Story 16.4] The header-derived tenant (X-EnGarde-Tenant-Id, set by the
    trusted proxy) is authoritative. Body/query-supplied tenant_id values are
    still accepted for caller compatibility but are never trusted: a claimed
    value that differs from the header is rejected.
    """
    if claimed_tenant_id and claimed_tenant_id != header_tenant_id:
        raise HTTPException(status_code=403, detail="tenant mismatch")
    return header_tenant_id
