from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: SecretStr
    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
