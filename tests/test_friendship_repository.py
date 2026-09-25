# tests/test_friendship_repository.py
"""
Tests de integración de FriendshipRepository (app/repositories/friendship_repository.py).

Usa SQLite en memoria (vía StaticPool para que todas las conexiones del
engine compartan la misma DB en memoria) en vez de mockear Session, para
probar las queries reales de SQLAlchemy.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base
from app.models.friendship import Friendship
from app.models.profile import Profile
from app.repositories.friendship_repository import FriendshipRepository


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def repository(db_session):
    return FriendshipRepository(db_session)


def _crear_perfil(db_session, **kwargs):
    defaults = dict(username="user", interests=[], activity_score=0, tags_agregados=[])
    defaults.update(kwargs)
    perfil = Profile(**defaults)
    db_session.add(perfil)
    db_session.commit()
    db_session.refresh(perfil)
    return perfil


class TestGetProfile:

    def test_devuelve_el_perfil_si_existe(self, db_session, repository):
        perfil = _crear_perfil(db_session, username="rocio")

        encontrado = repository.get_profile(perfil.id)

        assert encontrado is not None
        assert encontrado.id == perfil.id
        assert encontrado.username == "rocio"

    def test_devuelve_none_si_no_existe(self, repository):
        assert repository.get_profile(9999) is None


class TestGetAllFriendships:

    def test_lista_vacia_si_no_hay_friendships(self, repository):
        assert repository.get_all_friendships() == []

    def test_devuelve_todas_las_friendships_guardadas(self, db_session, repository):
        a = _crear_perfil(db_session, username="a")
        b = _crear_perfil(db_session, username="b")
        c = _crear_perfil(db_session, username="c")

        repository.save_prediction(a.id, b.id, 0.9)
        repository.save_prediction(a.id, c.id, 0.3)

        resultado = repository.get_all_friendships()

        assert len(resultado) == 2
        scores = {f.compatibility_score for f in resultado}
        assert scores == {0.9, 0.3}


class TestSavePrediction:

    def test_crea_y_persiste_el_registro(self, db_session, repository):
        a = _crear_perfil(db_session, username="a")
        b = _crear_perfil(db_session, username="b")

        registro = repository.save_prediction(a.id, b.id, 0.75)

        assert registro.id is not None
        assert registro.user_id == a.id
        assert registro.friend_id == b.id
        assert registro.compatibility_score == 0.75

    def test_el_registro_queda_realmente_en_la_db(self, db_session, repository):
        a = _crear_perfil(db_session, username="a")
        b = _crear_perfil(db_session, username="b")

        repository.save_prediction(a.id, b.id, 0.6)

        encontrado = (
            db_session.query(Friendship)
            .filter(Friendship.user_id == a.id, Friendship.friend_id == b.id)
            .first()
        )
        assert encontrado is not None
        assert encontrado.compatibility_score == 0.6

    def test_dos_predicciones_del_mismo_par_generan_dos_filas(
        self, db_session, repository
    ):
        """save_prediction no hace upsert: cada llamada agrega una fila nueva."""
        a = _crear_perfil(db_session, username="a")
        b = _crear_perfil(db_session, username="b")

        repository.save_prediction(a.id, b.id, 0.5)
        repository.save_prediction(a.id, b.id, 0.8)

        filas = (
            db_session.query(Friendship)
            .filter(Friendship.user_id == a.id, Friendship.friend_id == b.id)
            .all()
        )
        assert len(filas) == 2
