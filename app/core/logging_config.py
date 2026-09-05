"""Logging setup that avoids exposing candidate personal data."""

import logging


def configure_logging() -> None:
    """Configure concise application logs once at process startup."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
