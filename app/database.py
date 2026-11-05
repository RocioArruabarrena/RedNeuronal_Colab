from __future__ import annotations

from threading import Lock

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import settings


class Database:
    _instance: Database | None = None
    _lock = Lock()

    engine: Engine
    session_factory: sessionmaker

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.engine = create_engine(settings.database_url)
                    instance.session_factory = sessionmaker(
                        autocommit=False, autoflush=False, bind=instance.engine
                    )
                    cls._instance = instance
        return cls._instance


database = Database()
engine = database.engine
SessionLocal = database.session_factory
Base = declarative_base()


def get_db():
    db = database.session_factory()
    try:
        yield db
    finally:
        db.close()
