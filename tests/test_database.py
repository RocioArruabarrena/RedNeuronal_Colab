from app.database import Database, SessionLocal, database, engine


def test_database_usa_una_unica_instancia():
    assert Database() is database
    assert Database().engine is engine
    assert Database().session_factory is SessionLocal
