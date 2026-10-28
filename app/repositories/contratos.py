from abc import ABC, abstractmethod

from app.models.feedback import Feedback
from app.models.friendship import Friendship
from app.models.profile import Profile


class RepositorioAmistades(ABC):
    """Contrato para perfiles, amistades y predicciones."""

    @abstractmethod
    def get_profile(self, profile_id: int) -> Profile | None: ...

    @abstractmethod
    def get_all_friendships(self) -> list[Friendship]: ...

    @abstractmethod
    def save_prediction(
        self, user_id: int, friend_id: int, score: float
    ) -> Friendship: ...


class RepositorioFeedback(ABC):
    """Contrato para el feedback (valor real confirmado por el usuario)."""

    @abstractmethod
    def get_all_feedback(self) -> list[Feedback]: ...

    @abstractmethod
    def save_feedback(
        self,
        user_id: int,
        friend_id: int,
        features: list[float],
        predicted_score: float,
        real_label: bool,
    ) -> Feedback: ...


class RepositorioCompatibilidad(RepositorioAmistades, RepositorioFeedback):
    """Contrato completo: lo que necesita CompatibilityService."""
