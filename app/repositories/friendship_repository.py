from sqlalchemy.orm import Session
from app.models.feedback import FEATURE_COLUMNS, Feedback
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
        record = Friendship(
            user_id=user_id, friend_id=friend_id, compatibility_score=score
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def get_all_feedback(self) -> list[Feedback]:
        return self.db.query(Feedback).order_by(Feedback.id).all()

    def save_feedback(
        self,
        user_id: int,
        friend_id: int,
        features: list[float],
        predicted_score: float,
        real_label: bool,
    ) -> Feedback:
        record = Feedback(
            user_id=user_id,
            friend_id=friend_id,
            predicted_score=predicted_score,
            real_label=real_label,
            **dict(zip(FEATURE_COLUMNS, features)),
        )
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record
