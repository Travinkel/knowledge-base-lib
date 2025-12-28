"""
Knowledge Service - main facade for Knowledge Base operations.

Provides high-level operations for managing knowledge items,
atoms, and their relationships.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from ..models import (
    KnowledgeItem,
    KnowledgeEdge,
    LearningAtom,
    PlatinumAtom,
)
from ..storage import GraphStorage, VectorStorage


class KnowledgeService:
    """
    Main facade for Knowledge Base operations.

    Coordinates graph and vector storage to provide
    unified access to knowledge items and atoms.
    """

    def __init__(
        self,
        graph_storage: GraphStorage,
        vector_storage: Optional[VectorStorage] = None,
    ):
        self._graph = graph_storage
        self._vector = vector_storage

    # =========================================================
    # KNOWLEDGE ITEMS
    # =========================================================

    async def create_knowledge_item(
        self,
        item: KnowledgeItem,
        embed: bool = True,
    ) -> KnowledgeItem:
        """Create a knowledge item with optional embedding."""
        created = await self._graph.create_knowledge_item(item)

        if embed and self._vector and item.key_findings:
            # Generate and store embedding
            await self._vector.store_embedding(
                entity_id=created.id,
                embedding=[],  # Would come from embedding model
                entity_type="knowledge_item",
            )

        return created

    async def get_knowledge_item(self, item_id: UUID) -> Optional[KnowledgeItem]:
        """Get a knowledge item by ID."""
        return await self._graph.get_knowledge_item(item_id)

    async def find_by_reference(self, reference_id: str) -> Optional[KnowledgeItem]:
        """Find knowledge item by reference (DOI, OpenAlex ID, etc.)."""
        return await self._graph.get_knowledge_item_by_reference(reference_id)

    async def search_knowledge_items(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str] = None,
    ) -> List[KnowledgeItem]:
        """Semantic search for knowledge items."""
        if not self._vector:
            return []

        # Search by text embedding
        results = await self._vector.search_by_text(
            query_text=query,
            top_k=top_k,
            entity_type="knowledge_item",
        )

        # Fetch full items
        items = []
        for result in results:
            item = await self._graph.get_knowledge_item(result.entity_id)
            if item and (domain is None or item.domain == domain):
                items.append(item)

        return items

    # =========================================================
    # LEARNING ATOMS
    # =========================================================

    async def create_atom(
        self,
        atom: LearningAtom,
        embed: bool = True,
    ) -> LearningAtom:
        """Create a learning atom with optional embedding."""
        created = await self._graph.create_atom(atom)

        if embed and self._vector:
            content_text = str(atom.content)
            await self._vector.store_embedding(
                entity_id=created.id,
                embedding=[],  # Would come from embedding model
                entity_type="atom",
                metadata={"icap_level": atom.icap_level.value}
            )

        return created

    async def get_atom(self, atom_id: UUID) -> Optional[LearningAtom]:
        """Get a learning atom by ID."""
        return await self._graph.get_atom(atom_id)

    async def list_atoms(
        self,
        domain: Optional[str] = None,
        icap_level: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[LearningAtom]:
        """List atoms with optional filtering."""
        return await self._graph.list_atoms(
            domain=domain,
            icap_level=icap_level,
            limit=limit,
            offset=offset,
        )

    async def search_atoms(
        self,
        query: str,
        top_k: int = 10,
        icap_level: Optional[str] = None,
    ) -> List[LearningAtom]:
        """Semantic search for atoms."""
        if not self._vector:
            return []

        results = await self._vector.search_by_text(
            query_text=query,
            top_k=top_k,
            entity_type="atom",
        )

        atoms = []
        for result in results:
            atom = await self._graph.get_atom(result.entity_id)
            if atom and (icap_level is None or atom.icap_level.value == icap_level):
                atoms.append(atom)

        return atoms

    # =========================================================
    # PLATINUM ATOMS
    # =========================================================

    async def create_platinum_atom(
        self,
        platinum: PlatinumAtom,
    ) -> PlatinumAtom:
        """Create a Platinum-enriched atom."""
        created = await self._graph.create_platinum_atom(platinum)

        # Store enriched embedding if available
        if self._vector and platinum.embedding:
            await self._vector.store_embedding(
                entity_id=created.id,
                embedding=platinum.embedding,
                entity_type="platinum_atom",
                model=platinum.embedding_model or "unknown",
            )

        return created

    async def get_platinum_atom(self, atom_id: UUID) -> Optional[PlatinumAtom]:
        """Get a Platinum atom by ID."""
        return await self._graph.get_platinum_atom(atom_id)

    async def get_platinum_for_gold(self, gold_atom_id: UUID) -> Optional[PlatinumAtom]:
        """Get the Platinum enrichment for a Gold atom."""
        return await self._graph.get_platinum_atom_by_gold(gold_atom_id)

    # =========================================================
    # EDGES
    # =========================================================

    async def create_edge(self, edge: KnowledgeEdge) -> KnowledgeEdge:
        """Create a knowledge edge."""
        return await self._graph.create_edge(edge)

    async def get_edges(
        self,
        node_id: str,
        predicate: Optional[str] = None,
        direction: str = "both",
    ) -> List[KnowledgeEdge]:
        """Get edges for a node."""
        return await self._graph.get_edges(
            node_id=node_id,
            predicate=predicate,
            direction=direction,
        )

    # =========================================================
    # CONNECTION MANAGEMENT
    # =========================================================

    async def connect(self) -> None:
        """Connect to storage backends."""
        await self._graph.connect()
        if self._vector:
            await self._vector.connect()

    async def disconnect(self) -> None:
        """Disconnect from storage backends."""
        await self._graph.disconnect()
        if self._vector:
            await self._vector.disconnect()

    async def health_check(self) -> Dict[str, bool]:
        """Check health of storage backends."""
        health = {
            "graph": await self._graph.health_check(),
        }
        if self._vector:
            health["vector"] = await self._vector.health_check()
        return health
