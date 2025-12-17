from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.models import Base

# на всякий случай поддержим оба варианта названия поля в settings
POSTGRES_DSN = getattr(settings, "POSTGRES_DSN", None) or getattr(settings, "postgres_dsn")

engine = create_engine(POSTGRES_DSN, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
