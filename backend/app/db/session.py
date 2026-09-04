from collections.abc import Iterator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.settings import get_settings


def create_database_engine(database_url: str | None = None) -> Engine:
    """Create an engine for the configured database without opening a connection."""

    return create_engine(database_url or get_settings().database_url, pool_pre_ping=True)


def create_session_factory(database_url: str | None = None) -> sessionmaker[Session]:
    """Return a session factory bound to a newly constructed application engine."""

    return sessionmaker(
        bind=create_database_engine(database_url), autoflush=False, expire_on_commit=False
    )


def get_session() -> Iterator[Session]:
    """Provide an API-friendly transactional session scope."""

    session = create_session_factory()()
    try:
        yield session
    finally:
        session.close()
