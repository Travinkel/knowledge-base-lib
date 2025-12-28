"""
Graph storage interface for Knowledge Base.

Provides operations for storing and querying the knowledge graph,
including prerequisites, analogies, and evidence links.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..models import (
    KnowledgeItem,
    KnowledgeEdge,
    LearningAtom,
    PlatinumAtom,
    PrerequisiteLink,
    AnalogicalBridge,
)
from .base import StorageBackend


class GraphStorage(StorageBackend, ABC):
    """
    Abstract interface for graph storage operations.

    Implementations may use KuzuDB, Neo4j, PostgreSQL with ltree, etc.
    """

    # =========================================================
    # KNOWLEDGE ITEMS
    # =========================================================

    @abstractmethod
    async def create_knowledge_item(self, item: KnowledgeItem) -> KnowledgeItem:
        """Create a new knowledge item."""
        pass

    @abstractmethod
    async def get_knowledge_item(self, item_id: UUID) -> Optional[KnowledgeItem]:
        """Get a knowledge item by ID."""
        pass

    @abstractmethod
    async def get_knowledge_item_by_reference(self, reference_id: str) -> Optional[KnowledgeItem]:
        """Get a knowledge item by reference ID (DOI, etc.)."""
        pass

    @abstractmethod
    async def update_knowledge_item(self, item_id: UUID, updates: Dict[str, Any]) -> Optional[KnowledgeItem]:
        """Update a knowledge item."""
        pass

    @abstractmethod
    async def delete_knowledge_item(self, item_id: UUID) -> bool:
        """Delete a knowledge item."""
        pass

    # =========================================================
    # LEARNING ATOMS
    # =========================================================

    @abstractmethod
    async def create_atom(self, atom: LearningAtom) -> LearningAtom:
        """Create a new learning atom."""
        pass

    @abstractmethod
    async def get_atom(self, atom_id: UUID) -> Optional[LearningAtom]:
        """Get a learning atom by ID."""
        pass

    @abstractmethod
    async def list_atoms(
        self,
        domain: Optional[str] = None,
        icap_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LearningAtom]:
        """List learning atoms with optional filters."""
        pass

    # =========================================================
    # PLATINUM ATOMS
    # =========================================================

    @abstractmethod
    async def create_platinum_atom(self, atom: PlatinumAtom) -> PlatinumAtom:
        """Create a new Platinum-enriched atom."""
        pass

    @abstractmethod
    async def get_platinum_atom(self, atom_id: UUID) -> Optional[PlatinumAtom]:
        """Get a Platinum atom by ID."""
        pass

    @abstractmethod
    async def get_platinum_atom_by_gold(self, gold_atom_id: UUID) -> Optional[PlatinumAtom]:
        """Get the Platinum atom for a given Gold atom."""
        pass

    # =========================================================
    # PREREQUISITES
    # =========================================================

    @abstractmethod
    async def add_prerequisite(self, link: PrerequisiteLink) -> PrerequisiteLink:
        """Add a prerequisite relationship."""
        pass

    @abstractmethod
    async def get_prerequisites(
        self,
        atom_id: UUID,
        strength: Optional[str] = None,
    ) -> List[PrerequisiteLink]:
        """Get prerequisites for an atom."""
        pass

    @abstractmethod
    async def get_dependents(
        self,
        atom_id: UUID,
    ) -> List[PrerequisiteLink]:
        """Get atoms that depend on this atom as a prerequisite."""
        pass

    @abstractmethod
    async def remove_prerequisite(self, from_id: UUID, to_id: UUID) -> bool:
        """Remove a prerequisite link."""
        pass

    # =========================================================
    # ANALOGIES
    # =========================================================

    @abstractmethod
    async def add_analogy(self, bridge: AnalogicalBridge) -> AnalogicalBridge:
        """Add an analogical bridge."""
        pass

    @abstractmethod
    async def get_analogies(
        self,
        atom_id: UUID,
        bridge_type: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[AnalogicalBridge]:
        """Get analogies for an atom."""
        pass

    @abstractmethod
    async def remove_analogy(self, source_id: UUID, target_id: UUID) -> bool:
        """Remove an analogical bridge."""
        pass

    # =========================================================
    # EDGES (Generic)
    # =========================================================

    @abstractmethod
    async def create_edge(self, edge: KnowledgeEdge) -> KnowledgeEdge:
        """Create a generic knowledge edge."""
        pass

    @abstractmethod
    async def get_edges(
        self,
        node_id: str,
        predicate: Optional[str] = None,
        direction: str = "both",
        limit: int = 100,
    ) -> List[KnowledgeEdge]:
        """Get edges for a node."""
        pass

    # =========================================================
    # GRAPH QUERIES
    # =========================================================

    @abstractmethod
    async def find_path(
        self,
        from_id: UUID,
        to_id: UUID,
        max_depth: int = 5,
    ) -> Optional[List[UUID]]:
        """Find shortest path between two atoms."""
        pass

    @abstractmethod
    async def get_prerequisite_chain(
        self,
        atom_id: UUID,
        max_depth: int = 10,
    ) -> List[LearningAtom]:
        """Get the full prerequisite chain for an atom (topologically sorted)."""
        pass

    @abstractmethod
    async def detect_cycles(self) -> List[List[UUID]]:
        """Detect cycles in the prerequisite graph (should be empty for valid DAG)."""
        pass

    @abstractmethod
    async def topological_sort(
        self,
        atom_ids: Optional[List[UUID]] = None,
    ) -> List[UUID]:
        """Return atoms in learning order (prerequisites first)."""
        pass
