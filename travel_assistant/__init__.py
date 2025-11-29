"""
Enterprise Travel Assistant - A production-ready AI-powered travel planning system.

This package provides:
- Memory management with Mem0
- Semantic caching with Redis
- Request fingerprinting
- Gemini model comparison
- LangGraph workflow integration
- FastAPI REST API
- Beautiful UI/UX interface
"""

__version__ = "1.0.0"
__author__ = "Travel Assistant Team"
__email__ = "team@travelassistant.com"

from .core.config import settings
from .core.logger import get_logger

logger = get_logger(__name__)
logger.info(f"Travel Assistant v{__version__} initialized")

__all__ = [
    "settings",
    "get_logger",
]
