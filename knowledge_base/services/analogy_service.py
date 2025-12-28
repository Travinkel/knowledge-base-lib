"""
Analogy Service - manages analogical bridges between concepts.

Based on Structure-Mapping Theory (Gentner), provides operations
for detecting, storing, and querying analogical relationships.
"""

from typing import Any, Dict, List, Optional
from uuid import UUID

from ..models import AnalogicalBridge, LearningAtom
from ..storage import GraphStorage, VectorStorage


class AnalogyService:
    """
    Manages analogical bridges between atoms across domains.

    Uses both graph storage (explicit bridges) and vector similarity
    (implicit similarity) to find analogies.
    """

    def __init__(
        self,
        graph_storage: GraphStorage,
        vector_storage: Optional[VectorStorage] = None,
    ):
        self._graph = graph_storage
        self._vector = vector_storage

    # =========================================================
    # BRIDGE MANAGEMENT
    # =========================================================

    async def add_analogy(
        self,
        source_atom_id: UUID,
        target_atom_id: UUID,
        similarity_score: float,
        bridge_type: str = "structural",
        structural_mappings: Optional[List[Dict[str, str]]] = None,
        rationale: str = "",
    ) -> AnalogicalBridge:
        """
        Create an analogical bridge between two atoms.

        Args:
            source_atom_id: Source concept
            target_atom_id: Analogous concept
            similarity_score: How similar (0.0-1.0)
            bridge_type: structural|surface|mixed
            structural_mappings: List of {source: X, target: Y} mappings
            rationale: Explanation of the analogy

        Returns:
            Created bridge
        """
        bridge = AnalogicalBridge(
            source_atom_id=source_atom_id,
            target_atom_id=target_atom_id,
            similarity_score=similarity_score,
            bridge_type=bridge_type,
            structural_mappings=structural_mappings or [],
            rationale=rationale,
        )
        return await self._graph.add_analogy(bridge)

    async def remove_analogy(
        self,
        source_atom_id: UUID,
        target_atom_id: UUID,
    ) -> bool:
        """Remove an analogical bridge."""
        return await self._graph.remove_analogy(source_atom_id, target_atom_id)

    async def get_analogies(
        self,
        atom_id: UUID,
        bridge_type: Optional[str] = None,
        min_similarity: float = 0.0,
    ) -> List[AnalogicalBridge]:
        """
        Get analogies for an atom.

        Args:
            atom_id: The atom to get analogies for
            bridge_type: Filter by type (structural|surface|mixed)
            min_similarity: Minimum similarity threshold

        Returns:
            List of analogical bridges
        """
        return await self._graph.get_analogies(
            atom_id=atom_id,
            bridge_type=bridge_type,
            min_similarity=min_similarity,
        )

    # =========================================================
    # ANALOGY DISCOVERY
    # =========================================================

    async def find_similar_atoms(
        self,
        atom_id: UUID,
        top_k: int = 10,
        cross_domain: bool = True,
        min_similarity: float = 0.5,
    ) -> List[tuple[LearningAtom, float]]:
        """
        Find semantically similar atoms using vector search.

        Args:
            atom_id: The source atom
            top_k: Number of results
            cross_domain: If True, prioritize different domains
            min_similarity: Minimum similarity threshold

        Returns:
            List of (atom, similarity) tuples
        """
        if not self._vector:
            return []

        # Get source atom embedding
        embedding = await self._vector.get_embedding(atom_id)
        if not embedding:
            return []

        # Search for similar vectors
        results = await self._vector.search_similar(
            query_vector=embedding,
            top_k=top_k * 2,  # Get more, filter later
            min_similarity=min_similarity,
            entity_type="atom",
        )

        # Get source atom for domain filtering
        source_atom = await self._graph.get_atom(atom_id)
        source_domain = source_atom.content.get("domain") if source_atom else None

        # Fetch atoms and filter
        similar_atoms: List[tuple[LearningAtom, float]] = []
        for result in results:
            if result.entity_id == atom_id:
                continue  # Skip self

            atom = await self._graph.get_atom(result.entity_id)
            if not atom:
                continue

            # Cross-domain filtering
            atom_domain = atom.content.get("domain")
            if cross_domain and source_domain and atom_domain == source_domain:
                continue

            similar_atoms.append((atom, result.similarity))

        return similar_atoms[:top_k]

    async def suggest_analogies(
        self,
        atom_id: UUID,
        top_k: int = 5,
        min_structural_match: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Suggest potential analogies for review.

        Returns candidates with computed structural mappings
        for human or LLM review.
        """
        similar = await self.find_similar_atoms(
            atom_id=atom_id,
            top_k=top_k * 2,
            cross_domain=True,
            min_similarity=0.4,
        )

        source_atom = await self._graph.get_atom(atom_id)
        if not source_atom:
            return []

        suggestions = []
        for target_atom, similarity in similar:
            # Compute potential structural mappings
            mappings = self._compute_structural_mappings(source_atom, target_atom)

            if len(mappings) / max(1, self._count_features(source_atom)) >= min_structural_match:
                suggestions.append({
                    "source_atom_id": atom_id,
                    "target_atom_id": target_atom.id,
                    "target_atom": target_atom,
                    "similarity_score": similarity,
                    "structural_mappings": mappings,
                    "suggested_bridge_type": "structural" if len(mappings) > 2 else "surface",
                })

        return suggestions[:top_k]

    # =========================================================
    # ANALOGY APPLICATION
    # =========================================================

    async def get_bridging_atoms(
        self,
        from_domain: str,
        to_domain: str,
        limit: int = 5,
    ) -> List[AnalogicalBridge]:
        """
        Find atoms that bridge between two domains.

        Useful for scaffolding learning from familiar to unfamiliar domains.
        """
        # Get all atoms from source domain
        source_atoms = await self._graph.list_atoms(domain=from_domain, limit=100)

        bridges: List[AnalogicalBridge] = []
        for atom in source_atoms:
            analogies = await self._graph.get_analogies(atom.id)
            for bridge in analogies:
                target = await self._graph.get_atom(bridge.target_atom_id)
                if target and target.content.get("domain") == to_domain:
                    bridges.append(bridge)

        # Sort by similarity
        bridges.sort(key=lambda b: b.similarity_score, reverse=True)
        return bridges[:limit]

    # =========================================================
    # HELPER METHODS
    # =========================================================

    def _compute_structural_mappings(
        self,
        source: LearningAtom,
        target: LearningAtom,
    ) -> List[Dict[str, str]]:
        """
        Compute structural mappings between two atoms.

        Simple heuristic based on content structure overlap.
        For production, use LLM-based mapping.
        """
        mappings = []

        # Compare content keys
        source_content = source.content or {}
        target_content = target.content or {}

        for source_key in source_content:
            for target_key in target_content:
                # Simple key similarity check
                if self._keys_match(source_key, target_key):
                    mappings.append({
                        "source": source_key,
                        "target": target_key,
                    })

        return mappings

    def _keys_match(self, key1: str, key2: str) -> bool:
        """Check if two keys are semantically similar."""
        key1_lower = key1.lower()
        key2_lower = key2.lower()
        return (
            key1_lower == key2_lower or
            key1_lower in key2_lower or
            key2_lower in key1_lower
        )

    def _count_features(self, atom: LearningAtom) -> int:
        """Count structural features in an atom."""
        return len(atom.content or {})
