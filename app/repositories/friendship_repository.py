from sqlalchemy.orm import Session
from app.models.friendship import Friendship
from app.models.profile import Profile

class FriendshipRepository:
    """Patrón Repository: aísla el acceso a datos de friendships/profiles."""

    def __init__(self, db: Session):
        self.db = db

    def get_profile(self, profile_id: int) -> Profile | None:
        return self.db.query(Profile).filter(Profile.id == profile_id).first()

    def get_all_friendships(self) -> list[Friendship]:
        return self.db.query(Friendship).all()

    def save_prediction(self, user_id: int, friend_id: int, score: float) -> Friendship:
        record = Friendship(user_id=user_id, friend_id=friend_id, compatibility_score=score)
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record