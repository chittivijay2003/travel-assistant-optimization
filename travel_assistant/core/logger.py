"""
Enterprise Logging System
Provides comprehensive logging for file and console output.
"""

import os
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
from loguru import logger
from .config import settings


class TravelAssistantLogger:
    """Enterprise-grade logging system with structured logging."""

    def __init__(self):
        """Initialize the logging system."""
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Configure loguru logger with enterprise settings."""
        # Remove default logger
        logger.remove()

        # Console logging with colors and formatting
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=settings.log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # File logging with rotation and retention
        logger.add(
            settings.log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {extra} | {message}",
            level=settings.log_level,
            rotation=settings.log_rotation,
            retention=f"{settings.log_retention} days",
            compression="zip",
            backtrace=True,
            diagnose=True,
            serialize=False,
        )

        # JSON structured logging for production
        if settings.is_production:
            json_log_file = settings.log_file.replace(".log", "_structured.json")
            logger.add(
                json_log_file,
                format=self._json_formatter,
                level=settings.log_level,
                rotation=settings.log_rotation,
                retention=f"{settings.log_retention} days",
                compression="zip",
            )

    def _json_formatter(self, record: Any) -> str:
        """Format log record as JSON for structured logging."""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "function": record["function"],
            "line": record["line"],
            "message": record["message"],
            "extra": record.get("extra", {}),
            "exception": None,
        }

        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback.format(),
            }

        return json.dumps(log_entry, ensure_ascii=False)

    def log_request(
        self,
        method: str,
        endpoint: str,
        user_id: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None,
        fingerprint: Optional[str] = None,
    ) -> None:
        """Log incoming request with metadata."""
        extra_data = {
            "type": "request",
            "method": method,
            "endpoint": endpoint,
            "user_id": user_id or "anonymous",
            "fingerprint": fingerprint,
            "request_size": len(str(request_data)) if request_data else 0,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.bind(**extra_data).info(f"Incoming {method} request to {endpoint}")

    def log_response(
        self,
        endpoint: str,
        status_code: int,
        response_time_ms: float,
        cache_hit: bool = False,
        model_used: Optional[str] = None,
    ) -> None:
        """Log response with performance metrics."""
        extra_data = {
            "type": "response",
            "endpoint": endpoint,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
            "cache_hit": cache_hit,
            "model_used": model_used,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.bind(**extra_data).info(
            f"Response {status_code} in {response_time_ms:.2f}ms "
            f"{'(cached)' if cache_hit else '(fresh)'}"
        )

    def log_memory_operation(
        self,
        operation: str,
        user_id: str,
        success: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log memory operations (Mem0)."""
        extra_data = {
            "type": "memory",
            "operation": operation,
            "user_id": user_id,
            "success": success,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        level = "info" if success else "warning"
        message = f"Memory {operation} {'succeeded' if success else 'failed'} for user {user_id}"

        getattr(logger.bind(**extra_data), level)(message)

    def log_cache_operation(
        self,
        operation: str,
        cache_key: str,
        hit: Optional[bool] = None,
        similarity_score: Optional[float] = None,
    ) -> None:
        """Log cache operations."""
        extra_data = {
            "type": "cache",
            "operation": operation,
            "cache_key": cache_key[:50] + "..." if len(cache_key) > 50 else cache_key,
            "hit": hit,
            "similarity_score": similarity_score,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if hit is not None:
            message = f"Cache {operation} - {'HIT' if hit else 'MISS'}"
            if similarity_score:
                message += f" (similarity: {similarity_score:.3f})"
        else:
            message = f"Cache {operation}"

        logger.bind(**extra_data).info(message)

    def log_model_comparison(
        self, query: str, flash_metrics: Dict[str, Any], pro_metrics: Dict[str, Any]
    ) -> None:
        """Log Gemini model comparison results."""
        extra_data = {
            "type": "model_comparison",
            "query_length": len(query),
            "flash_metrics": flash_metrics,
            "pro_metrics": pro_metrics,
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.bind(**extra_data).info(
            f"Model comparison: Flash {flash_metrics.get('latency_ms', 0):.0f}ms "
            f"vs Pro {pro_metrics.get('latency_ms', 0):.0f}ms"
        )

    def log_fingerprint(
        self,
        original_query: str,
        fingerprint: str,
        similar_found: bool,
        similarity_score: Optional[float] = None,
    ) -> None:
        """Log fingerprinting operations."""
        extra_data = {
            "type": "fingerprint",
            "query_length": len(original_query),
            "fingerprint": fingerprint,
            "similar_found": similar_found,
            "similarity_score": similarity_score,
            "timestamp": datetime.utcnow().isoformat(),
        }

        message = f"Fingerprint generated: {fingerprint[:16]}..."
        if similar_found:
            message += f" (similar query found, score: {similarity_score:.3f})"

        logger.bind(**extra_data).info(message)

    def log_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Log errors with full context."""
        extra_data = {
            "type": "error",
            "error_type": type(error).__name__,
            "user_id": user_id or "unknown",
            "context": context or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        logger.bind(**extra_data).error(f"Error occurred: {str(error)}")


# Global logger instance
travel_logger = TravelAssistantLogger()


def get_logger(name: str = __name__):
    """Get a logger instance for the given module."""
    return logger.bind(module=name)


# Convenience functions for common logging patterns
def log_info(message: str, **kwargs):
    """Log info message with optional extra data."""
    logger.bind(**kwargs).info(message)


def log_warning(message: str, **kwargs):
    """Log warning message with optional extra data."""
    logger.bind(**kwargs).warning(message)


def log_error(message: str, **kwargs):
    """Log error message with optional extra data."""
    logger.bind(**kwargs).error(message)


def log_debug(message: str, **kwargs):
    """Log debug message with optional extra data."""
    logger.bind(**kwargs).debug(message)
