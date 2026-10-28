from pydantic import BaseModel, Field


class CompatibilityRequest(BaseModel):
    user_id: int = Field(..., examples=[1])
    friend_id: int = Field(..., examples=[2])


class CompatibilityResponse(BaseModel):
    user_id: int
    friend_id: int
    compatibility_score: float
    is_compatible: bool


class FeedbackRequest(BaseModel):
    user_id: int = Field(..., examples=[1])
    friend_id: int = Field(..., examples=[2])
    real_label: bool = Field(
        ..., description="Valor real: True si son compatibles", examples=[True]
    )


class FeedbackResponse(BaseModel):
    id: int
    user_id: int
    friend_id: int
    predicted_score: float
    real_label: bool
    features: dict[str, float]


class MetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float | None = None


class RetrainResponse(BaseModel):
    trained_at: str
    feedback_count: int
    train_size: int
    test_before: MetricsResponse
    test_after: MetricsResponse
    feedback_before: MetricsResponse
    feedback_after: MetricsResponse
