# from pydantic import BaseSettings, Field
from pydantic_settings import BaseSettings
from pydantic import Field # Field is still in pydantic

from typing import Optional, List
import os
from pathlib import Path

class Settings(BaseSettings):
    # App settings
    app_name: str = "Music Recommender API"
    version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server settings
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8000, env="PORT")
    
    # Database settings
    database_url: str = Field(env="DATABASE_URL")
    database_echo: bool = Field(default=False, env="DATABASE_ECHO")
    
    # Data source settings
    data_source: str = Field(default="csv", env="DATA_SOURCE")  # "csv" or "db"
    csv_file_path: str = Field(default="data/dataset.csv", env="CSV_FILE_PATH")
    max_csv_rows: Optional[int] = Field(default=20000, env="MAX_CSV_ROWS")
    
    # CORS settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS"
    )
    
    # ML Model settings
    similarity_features: List[str] = Field(
        default=[
            'acousticness', 'danceability', 'energy', 'instrumentalness',
            'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'popularity'
        ]
    )
    
    # File upload settings
    upload_dir: str = Field(default="temp_uploads", env="UPLOAD_DIR")
    max_file_size: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Ensure upload directory exists
Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)