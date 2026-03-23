from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError

from apps.core.config import Settings, get_settings
from apps.core.security import decoded_access_token

bearer_schema = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_schema),
                     settings: Settings = Depends(get_settings)):
    token = credentials.credentials
    try:
        payload = decoded_access_token(token, settings)
        username = payload.get("sub")

        if not username:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Invalid token payload")
        return {"username": username}
    except JWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired token") from exc
