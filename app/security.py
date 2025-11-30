from fastapi import Header, HTTPException
from app.settings import settings


def require_api_key(x_api_key: str | None = Header(default=None, alias="X-Api-Key")):
    if not x_api_key or x_api_key != settings.GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="invalid X-Api-Key")
    return True

