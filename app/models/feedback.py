from sqlalchemy import Column, Integer, Boolean, DateTime, ForeignKey, Float
from sqlalchemy.sql import func
from app.database import Base

# Mismo orden que el vector de entrada de CompatibilityNet
FEATURE_COLUMNS = [
    "similitud_intereses",
    "diff_activity",
    "similitud_tags_posts",
    "tiene_tags_posts",
    "mismo_avatar_subcultura",
]


class Feedback(Base):
    """Par (features, etiqueta real) confirmado por el usuario.

    Es la fuente de datos para reentrenar la red.
    """

    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    friend_id = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    similitud_intereses = Column(Float, nullable=False)
    diff_activity = Column(Float, nullable=False)
    similitud_tags_posts = Column(Float, nullable=False)
    tiene_tags_posts = Column(Float, nullable=False)
    mismo_avatar_subcultura = Column(Float, nullable=False)
    predicted_score = Column(Float, nullable=False)
    real_label = Column(Boolean, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())