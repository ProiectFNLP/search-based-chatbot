from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Optional

class Settings(BaseSettings):
    openai_api_key: SecretStr | None = None

    class Config:
        env_file = ".env"



settings = Settings()
