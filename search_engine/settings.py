from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    openai_api_key: SecretStr
    flan_t5_model_path: Optional[str] = None
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
