from pydantic import BaseModel, Field


class CompatibilityRequest(BaseModel):
    user_id: int = Field(..., examples=[1])
    friend_id: int = Field(..., examples=[2])


class CompatibilityResponse(BaseModel):
    user_id: int
    friend_id: int
    compatibility_score: float
    is_compatible: bool
