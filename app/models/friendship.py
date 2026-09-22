from sqlalchemy import Column, Integer, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
from app.database import Base

class Friendship(Base):
    __tablename__ = "friendships"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    is_mutual = Column(Boolean, default=False)
    compatibility_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())