from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, create_engine


def make_engine(url: str) -> Engine:
    if not url.startswith("sqlite"):
        return create_engine(url)
    kwargs: dict = {"connect_args": {"check_same_thread": False}}
    if url in ("sqlite://", "sqlite:///:memory:"):
        kwargs["poolclass"] = StaticPool  # one shared in-memory database (tests)
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> None:
    from icp_api import models  # noqa: F401  (registers the tables)

    SQLModel.metadata.create_all(engine)
