"""
Atom storage with PostgreSQL persistence.

Implements CRUD operations for learning atoms with:
- Connection pooling via SQLAlchemy
- Input validation
- Audit logging
"""

import logging
import os
from datetime import datetime
from typing import Protocol

from .models import Atom, AtomLayer


logger = logging.getLogger(__name__)


class AtomStoreError(Exception):
    """Base exception for atom store operations."""
    pass


class AtomNotFoundError(AtomStoreError):
    """Raised when an atom is not found."""
    pass


class AtomValidationError(AtomStoreError):
    """Raised when atom validation fails."""
    pass


class AtomStore(Protocol):
    """Protocol for atom storage backends."""

    def create(self, atom: Atom) -> Atom: ...
    def get(self, atom_id: str) -> Atom | None: ...
    def update(self, atom: Atom) -> Atom: ...
    def delete(self, atom_id: str) -> bool: ...
    def list_by_layer(self, layer: AtomLayer, limit: int = 100) -> list[Atom]: ...


class InMemoryAtomStore:
    """
    In-memory atom store for testing and development.

    Note: This is NOT for production. Use PostgresAtomStore instead.
    """

    def __init__(self) -> None:
        self._atoms: dict[str, Atom] = {}

    def create(self, atom: Atom) -> Atom:
        """Create a new atom."""
        self._validate_atom(atom)
        if atom.id in self._atoms:
            raise AtomValidationError(f"Atom already exists: {atom.id}")
        self._atoms[atom.id] = atom
        logger.info(f"Created atom: {atom.id} ({atom.layer.value})")
        return atom

    def get(self, atom_id: str) -> Atom | None:
        """Get an atom by ID."""
        return self._atoms.get(atom_id)

    def update(self, atom: Atom) -> Atom:
        """Update an existing atom."""
        self._validate_atom(atom)
        if atom.id not in self._atoms:
            raise AtomNotFoundError(f"Atom not found: {atom.id}")
        atom.updated_at = datetime.utcnow()
        self._atoms[atom.id] = atom
        logger.info(f"Updated atom: {atom.id}")
        return atom

    def delete(self, atom_id: str) -> bool:
        """Delete an atom by ID."""
        if atom_id not in self._atoms:
            return False
        del self._atoms[atom_id]
        logger.info(f"Deleted atom: {atom_id}")
        return True

    def list_by_layer(self, layer: AtomLayer, limit: int = 100) -> list[Atom]:
        """List atoms by layer."""
        return [
            a for a in list(self._atoms.values())[:limit]
            if a.layer == layer
        ]

    def _validate_atom(self, atom: Atom) -> None:
        """Validate atom before storage."""
        if not atom.id:
            raise AtomValidationError("Atom ID is required")
        if not atom.content:
            raise AtomValidationError("Atom content is required")
        if not isinstance(atom.layer, AtomLayer):
            raise AtomValidationError("Atom layer must be an AtomLayer enum")


class PostgresAtomStore:
    """
    PostgreSQL-backed atom store.

    Uses SQLAlchemy for connection pooling and ORM operations.
    Connection string from environment: ASTARTES_KB_CONNECTION
    """

    def __init__(self, connection_string: str | None = None) -> None:
        self._connection_string = connection_string or os.getenv("ASTARTES_KB_CONNECTION")
        if not self._connection_string:
            raise AtomStoreError(
                "Database connection required. Set ASTARTES_KB_CONNECTION env var "
                "or pass connection_string parameter."
            )
        self._engine = None
        self._session_factory = None

    def _get_engine(self):
        """Lazy initialization of SQLAlchemy engine with connection pooling."""
        if self._engine is None:
            try:
                from sqlalchemy import create_engine
                from sqlalchemy.pool import QueuePool

                self._engine = create_engine(
                    self._connection_string,
                    poolclass=QueuePool,
                    pool_size=5,
                    max_overflow=10,
                    pool_pre_ping=True,  # Verify connections before use
                )
                logger.info("PostgreSQL connection pool initialized")
            except ImportError:
                raise AtomStoreError(
                    "SQLAlchemy not installed. Run: pip install sqlalchemy psycopg2-binary"
                )
        return self._engine

    def create(self, atom: Atom) -> Atom:
        """Create a new atom in PostgreSQL."""
        # TODO: Implement with SQLAlchemy ORM
        raise NotImplementedError("PostgreSQL store not yet implemented")

    def get(self, atom_id: str) -> Atom | None:
        """Get an atom by ID from PostgreSQL."""
        # TODO: Implement with SQLAlchemy ORM
        raise NotImplementedError("PostgreSQL store not yet implemented")

    def update(self, atom: Atom) -> Atom:
        """Update an existing atom in PostgreSQL."""
        # TODO: Implement with SQLAlchemy ORM
        raise NotImplementedError("PostgreSQL store not yet implemented")

    def delete(self, atom_id: str) -> bool:
        """Delete an atom from PostgreSQL."""
        # TODO: Implement with SQLAlchemy ORM
        raise NotImplementedError("PostgreSQL store not yet implemented")

    def list_by_layer(self, layer: AtomLayer, limit: int = 100) -> list[Atom]:
        """List atoms by layer from PostgreSQL."""
        # TODO: Implement with SQLAlchemy ORM
        raise NotImplementedError("PostgreSQL store not yet implemented")
