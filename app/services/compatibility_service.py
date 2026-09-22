import torch

from app.repositories.friendship_repository import FriendshipRepository
from app.ml.model_factory import ModelFactory
from app.config import settings


def _extraer_subcultura(intereses: list[str] | None) -> str | None:
    if not intereses:
        return None
    for tag in intereses:
        if isinstance(tag, str) and tag.startswith("subcultura:"):
            return tag.split(":", 1)[1]
    return None


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _calcular_features(perfil_a, perfil_b) -> list[float]:
    intereses_a = set(perfil_a.interests or [])
    intereses_b = set(perfil_b.interests or [])
    similitud_intereses = _jaccard(intereses_a, intereses_b)

    diff_activity = abs((perfil_a.activity_score or 0) - (perfil_b.activity_score or 0))

    tags_posts_a = set(perfil_a.tags_agregados or [])
    tags_posts_b = set(perfil_b.tags_agregados or [])
    similitud_tags_posts = _jaccard(tags_posts_a, tags_posts_b)
    tiene_tags_posts = 1.0 if tags_posts_a and tags_posts_b else 0.0

    sub_a = _extraer_subcultura(perfil_a.interests)
    sub_b = _extraer_subcultura(perfil_b.interests)
    mismo_avatar_subcultura = 1.0 if sub_a and sub_b and sub_a == sub_b else 0.0

    return [
        similitud_intereses,
        diff_activity,
        similitud_tags_posts,
        tiene_tags_posts,
        mismo_avatar_subcultura,
    ]

class CompatibilityService:
    def __init__(self, repository: FriendshipRepository):
        self.repository = repository
        self.model = ModelFactory.get_model(settings.model_path)

    def predict(self, user_id: int, friend_id: int) -> dict:
        user = self.repository.get_profile(user_id)
        friend = self.repository.get_profile(friend_id)

        if not user or not friend:
            raise ValueError("Perfil de usuario o amigo no encontrado")

        features = self._build_features(user, friend)
        features_tensor = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            score = float(self.model(features_tensor).item())

        self.repository.save_prediction(user_id, friend_id, score)

        return {
            "user_id": user_id,
            "friend_id": friend_id,
            "compatibility_score": round(score, 4),
            "is_compatible": score >= 0.5,
        }

    def _build_features(self, user, friend) -> list[float]:
        return _calcular_features(user, friend)