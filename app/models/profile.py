from sqlalchemy import Column, Integer, String, JSON
from app.database import Base

class Profile(Base):
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    interests = Column(JSON, nullable=True)   # lista de intereses/tags
    activity_score = Column(Integer, default=0)
    tags_agregados = Column(JSON, nullable=True)  # union de tags de todos los posts públicos del usuario