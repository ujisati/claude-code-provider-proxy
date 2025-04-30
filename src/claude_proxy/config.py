from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file="../../.env")

    openrouter_api_key: str
    big_model_name: str
    small_model_name: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    referrer_url: str = "http://localhost:8082/ClaudeProxy"
    max_tokens: int = 4096

    app_name: str = "ClaudeProxy"
    app_version: str = "0.1"
    log_level: str = "INFO"
    log_file_path: str = "log.jsonl"
    host: str = "127.0.0.1"
    port: int = 8082
    reload: bool = True


settings = Settings()
