"""
Centralized structured logging configuration.

Compatible with:
- FastAPI
- Uvicorn
- Structlog
- Enterprise observability pipelines
"""

import logging
import sys

import structlog


def configure_logger():

    # =========================================================
    # Shared Logging Configuration
    # =========================================================

    timestamper = structlog.processors.TimeStamper(fmt="iso")

    shared_processors = [
        structlog.stdlib.add_log_level,
        timestamper,
    ]

    # =========================================================
    # Configure Standard Logging
    # =========================================================

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.INFO,
    )

    # =========================================================
    # Configure Structlog
    # =========================================================

    structlog.configure(

        processors=[
            *shared_processors,

            # Render readable logs locally
            structlog.dev.ConsoleRenderer()
        ],

        context_class=dict,

        logger_factory=structlog.stdlib.LoggerFactory(),

        wrapper_class=structlog.stdlib.BoundLogger,

        cache_logger_on_first_use=True,
    )

    # =========================================================
    # IMPORTANT:
    # Use named logger for FastAPI/Uvicorn compatibility
    # =========================================================

    return structlog.get_logger("rag-app")


logger = configure_logger()