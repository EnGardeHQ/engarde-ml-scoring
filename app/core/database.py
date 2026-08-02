from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import settings

engine = create_engine(
    settings.database_url or "postgresql://localhost/wallet_dev",
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    connect_args={"options": f"-csearch_path={settings.wallet_db_schema},public"},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
