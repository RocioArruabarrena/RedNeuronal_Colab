from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.repositories.friendship_repository import FriendshipRepository
from app.services.compatibility_service import CompatibilityService
from app.views.compatibility_schema import CompatibilityRequest, CompatibilityResponse

router = APIRouter(prefix="/compatibility", tags=["Compatibility"])

def get_service(db: Session = Depends(get_db)) -> CompatibilityService:
    repository = FriendshipRepository(db)
    return CompatibilityService(repository)

@router.post("/predict", response_model=CompatibilityResponse)
def predict_compatibility(
    request: CompatibilityRequest,
    service: CompatibilityService = Depends(get_service),
):
    try:
        return service.predict(request.user_id, request.friend_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))