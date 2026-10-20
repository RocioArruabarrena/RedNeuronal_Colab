from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.repositories.friendship_repository import FriendshipRepository
from app.services.compatibility_service import CompatibilityService
from app.services.metrics_service import MetricsService
from app.services.retrain_service import RetrainService
from app.views.compatibility_schema import (
    CompatibilityRequest,
    CompatibilityResponse,
    FeedbackRequest,
    FeedbackResponse,
    RetrainResponse,
)

router = APIRouter(prefix="/compatibility", tags=["Compatibility"])


def get_service(db: Session = Depends(get_db)) -> CompatibilityService:
    repository = FriendshipRepository(db)
    return CompatibilityService(repository)


def get_retrain_service(db: Session = Depends(get_db)) -> RetrainService:
    return RetrainService(FriendshipRepository(db))


def get_metrics_service() -> MetricsService:
    return MetricsService()


@router.post("/predict", response_model=CompatibilityResponse)
def predict_compatibility(
    request: CompatibilityRequest,
    service: CompatibilityService = Depends(get_service),
):
    try:
        return service.predict(request.user_id, request.friend_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/feedback", response_model=FeedbackResponse, status_code=201)
def save_feedback(
    request: FeedbackRequest,
    service: CompatibilityService = Depends(get_service),
):
    try:
        return service.save_feedback(
            request.user_id, request.friend_id, request.real_label
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/retrain", response_model=RetrainResponse)
def retrain_model(service: RetrainService = Depends(get_retrain_service)):
    try:
        return service.retrain()
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics", response_model=RetrainResponse)
def get_metrics(service: MetricsService = Depends(get_metrics_service)):
    try:
        return service.get_last_metrics()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))