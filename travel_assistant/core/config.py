"""
Enterprise Configuration Management
Handles all application settings with environment variable support.
"""

import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Enterprise-grade configuration management."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Application Settings
    app_name: str = Field(default="travel-assistant-optimization")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    # API Configuration
    api_timeout: int = Field(default=30)
    api_base_url: str = Field(default="http://localhost:8000")

    # Google Gemini API Configuration
    google_api_key: str = Field(default="")
    gemini_flash_model: str = Field(default="gemini-1.5-flash")
    gemini_pro_model: str = Field(default="gemini-1.5-pro")

    # External API Keys
    openweather_api_key: str = Field(default="")
    google_maps_api_key: str = Field(default="")

    # Redis Configuration
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    redis_url: str = Field(default="redis://localhost:6379/0")

    # Memory Configuration (Mem0)
    mem0_api_key: str = Field(default="")
    mem0_user_id: str = Field(default="default_user")
    mem0_organization_name: str = Field(default="travel_assistant")

    # Semantic Cache Configuration
    cache_ttl: int = Field(default=3600)
    cache_similarity_threshold: float = Field(default=0.85)
    semantic_cache_enabled: bool = Field(default=True)

    # Request Fingerprinting
    fingerprint_algorithm: str = Field(default="sha256")
    fingerprint_salt: str = Field(default="default_salt")

    # File Storage & Logging
    upload_folder: str = Field(default="uploads/")
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/travel_assistant.log")
    log_rotation: str = Field(default="10MB")
    log_retention: int = Field(default=30)

    # Travel-specific Configuration
    default_currency: str = Field(default="USD")
    default_language: str = Field(default="en")
    max_destinations_per_trip: int = Field(default=10)
    search_radius_km: int = Field(default=50)

    # Performance & Security
    max_request_size: int = Field(default=10485760)
    rate_limit_requests: int = Field(default=100)
    rate_limit_window: int = Field(default=3600)
    jwt_secret_key: str = Field(default="default_jwt_secret")
    session_timeout: int = Field(default=7200)

    # UI/UX Configuration
    ui_theme: str = Field(default="light")
    enable_metrics_dashboard: bool = Field(default=True)
    enable_chat_interface: bool = Field(default=True)
    chat_history_limit: int = Field(default=50)

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == "development"

    def validate_required_keys(self) -> list[str]:
        """Validate required API keys and return missing ones."""
        missing_keys = []

        if not self.google_api_key:
            missing_keys.append("GOOGLE_API_KEY")

        # Add other required key validations as needed
        return missing_keys

    def get_redis_config(self) -> dict:
        """Get Redis configuration dictionary."""
        return {
            "host": self.redis_host,
            "port": self.redis_port,
            "db": self.redis_db,
            "password": self.redis_password if self.redis_password else None,
            "decode_responses": True,
        }


# Global settings instance
settings = Settings()

# Ensure required directories exist
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)
os.makedirs(settings.upload_folder, exist_ok=True)
