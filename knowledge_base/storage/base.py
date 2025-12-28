"""
Base storage interface for Knowledge Base.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar
from uuid import UUID

T = TypeVar("T")


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Implementations may use PostgreSQL, KuzuDB, Redis, etc.
    """

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to storage backend."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to storage backend."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if storage is healthy and accessible."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if currently connected."""
        pass
