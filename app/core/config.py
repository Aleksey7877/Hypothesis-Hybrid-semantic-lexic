from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Читаем настройки из ENV (docker-compose environment).
    .env рядом с compose может быть, но не обязателен.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- app ---
    APP_NAME: str = "aedev-rag"
    LOG_LEVEL: str = "INFO"

    # --- retrieval params ---
    K_VEC: int = 3
    K_LEX: int = 3
    RRF_K0: int = 60
    PARENT_ALPHA: float = 0.25

    # --- chunk sizes ---
    PARENT_CHARS: int = 2500
    CHILD_CHARS: int = 300

    # --- external services ---
    POSTGRES_DSN: str = "postgresql+psycopg://postgres:postgres@postgres:5432/aedev"
    QDRANT_URL: str = "http://qdrant:6333"
    ELASTIC_URL: str = "http://elasticsearch:9200"


settings = Settings()
