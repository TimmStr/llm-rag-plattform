from fastapi import APIRouter, HTTPException, status, Depends

from apps.api.schemas.auth import LoginRequest, TokenResponse
from apps.core.config import get_settings, Settings
from apps.core.security import authenticate_demo_user, create_access_token

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(request: LoginRequest,
          settings: Settings = Depends(get_settings)) -> TokenResponse:
    is_valid = authenticate_demo_user(
        username=request.username,
        password=request.password,
        settings=settings
    )
    if not is_valid:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid credentials")
    token = create_access_token(
        data={"sub": request.username},
        settings=settings
    )
    return TokenResponse(access_token=token)
