"""
Search Service - unified search across knowledge base.

Provides vector-based semantic search, keyword search,
and hybrid search combining both approaches.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from ..models import KnowledgeItem, LearningAtom, PlatinumAtom
from ..storage import GraphStorage, VectorStorage, SimilarityResult


@dataclass
class SearchResult:
    """Unified search result."""
    entity_id: UUID
    entity_type: str  # knowledge_item, atom, platinum_atom
    score: float
    entity: Optional[Union[KnowledgeItem, LearningAtom, PlatinumAtom]] = None
    highlights: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class SearchService:
    """
    Unified search service for the Knowledge Base.

    Combines vector similarity search with graph traversal
    for comprehensive knowledge retrieval.
    """

    def __init__(
        self,
        graph_storage: GraphStorage,
        vector_storage: Optional[VectorStorage] = None,
    ):
        self._graph = graph_storage
        self._vector = vector_storage

    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================

    async def semantic_search(
        self,
        query: str,
        top_k: int = 10,
        entity_types: Optional[List[str]] = None,
        min_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """
        Semantic search using vector embeddings.

        Args:
            query: Natural language query
            top_k: Number of results
            entity_types: Filter by type (knowledge_item, atom, platinum_atom)
            min_score: Minimum similarity score
            filters: Additional metadata filters

        Returns:
            List of search results with entities
        """
        if not self._vector:
            return []

        entity_types = entity_types or ["atom", "knowledge_item", "platinum_atom"]
        all_results: List[SearchResult] = []

        for entity_type in entity_types:
            results = await self._vector.search_by_text(
                query_text=query,
                top_k=top_k,
                min_similarity=min_score,
                entity_type=entity_type,
            )

            for result in results:
                entity = await self._fetch_entity(result.entity_id, entity_type)
                if entity and self._matches_filters(entity, filters):
                    all_results.append(SearchResult(
                        entity_id=result.entity_id,
                        entity_type=entity_type,
                        score=result.similarity,
                        entity=entity,
                        metadata=result.metadata,
                    ))

        # Sort by score and limit
        all_results.sort(key=lambda r: r.score, reverse=True)
        return all_results[:top_k]

    async def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        entity_type: Optional[str] = None,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Direct vector similarity search.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results
            entity_type: Filter by entity type
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        if not self._vector:
            return []

        results = await self._vector.search_similar(
            query_vector=query_vector,
            top_k=top_k,
            min_similarity=min_score,
            entity_type=entity_type,
        )

        search_results = []
        for result in results:
            entity = await self._fetch_entity(
                result.entity_id,
                entity_type or "atom",
            )
            search_results.append(SearchResult(
                entity_id=result.entity_id,
                entity_type=entity_type or "atom",
                score=result.similarity,
                entity=entity,
                metadata=result.metadata,
            ))

        return search_results

    # =========================================================
    # HYBRID SEARCH
    # =========================================================

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.7,  # Weight for semantic vs. graph
        include_prerequisites: bool = False,
        include_analogies: bool = False,
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and graph-based retrieval.

        Args:
            query: Natural language query
            top_k: Number of results
            alpha: Balance between semantic (1.0) and graph (0.0)
            include_prerequisites: Expand results with prerequisites
            include_analogies: Expand results with analogies

        Returns:
            Combined search results
        """
        # Get semantic results
        semantic_results = await self.semantic_search(
            query=query,
            top_k=top_k * 2,
            entity_types=["atom"],
        )

        # Build result set with scores
        result_map: Dict[UUID, SearchResult] = {}
        for result in semantic_results:
            result_map[result.entity_id] = result
            result.score = result.score * alpha

        # Expand with graph traversal
        if include_prerequisites or include_analogies:
            seed_ids = list(result_map.keys())[:top_k // 2]

            for seed_id in seed_ids:
                # Get prerequisites
                if include_prerequisites:
                    prereqs = await self._graph.get_prerequisites(seed_id)
                    for link in prereqs:
                        if link.to_atom_id not in result_map:
                            atom = await self._graph.get_atom(link.to_atom_id)
                            if atom:
                                # Score based on link strength and confidence
                                prereq_score = link.confidence * (1 - alpha) * 0.8
                                result_map[link.to_atom_id] = SearchResult(
                                    entity_id=link.to_atom_id,
                                    entity_type="atom",
                                    score=prereq_score,
                                    entity=atom,
                                    metadata={"source": "prerequisite"},
                                )

                # Get analogies
                if include_analogies:
                    analogies = await self._graph.get_analogies(seed_id)
                    for bridge in analogies:
                        if bridge.target_atom_id not in result_map:
                            atom = await self._graph.get_atom(bridge.target_atom_id)
                            if atom:
                                analogy_score = bridge.similarity_score * (1 - alpha) * 0.7
                                result_map[bridge.target_atom_id] = SearchResult(
                                    entity_id=bridge.target_atom_id,
                                    entity_type="atom",
                                    score=analogy_score,
                                    entity=atom,
                                    metadata={"source": "analogy"},
                                )

        # Sort and return
        results = list(result_map.values())
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # =========================================================
    # SPECIALIZED SEARCHES
    # =========================================================

    async def find_atoms_for_concept(
        self,
        concept_query: str,
        icap_level: Optional[str] = None,
        difficulty_range: Optional[tuple[float, float]] = None,
        limit: int = 10,
    ) -> List[LearningAtom]:
        """
        Find learning atoms matching a concept query.

        Args:
            concept_query: Natural language concept description
            icap_level: Filter by ICAP level
            difficulty_range: (min, max) difficulty filter
            limit: Maximum results

        Returns:
            Matching atoms
        """
        results = await self.semantic_search(
            query=concept_query,
            top_k=limit * 2,
            entity_types=["atom"],
        )

        atoms = []
        for result in results:
            if not isinstance(result.entity, LearningAtom):
                continue

            atom = result.entity

            # Apply filters
            if icap_level and atom.icap_level.value != icap_level:
                continue

            if difficulty_range and atom.difficulty is not None:
                min_diff, max_diff = difficulty_range
                if not (min_diff <= atom.difficulty <= max_diff):
                    continue

            atoms.append(atom)
            if len(atoms) >= limit:
                break

        return atoms

    async def find_evidence(
        self,
        claim: str,
        min_confidence: float = 0.5,
        study_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[KnowledgeItem]:
        """
        Find evidence supporting or related to a claim.

        Args:
            claim: The claim to find evidence for
            min_confidence: Minimum confidence threshold
            study_types: Filter by study type (rct, meta_analysis, etc.)
            limit: Maximum results

        Returns:
            Matching knowledge items
        """
        results = await self.semantic_search(
            query=claim,
            top_k=limit * 2,
            entity_types=["knowledge_item"],
        )

        items = []
        for result in results:
            if not isinstance(result.entity, KnowledgeItem):
                continue

            item = result.entity

            # Apply filters
            if item.adjusted_confidence is not None:
                if item.adjusted_confidence < min_confidence:
                    continue

            if study_types and item.study_type:
                if item.study_type.value not in study_types:
                    continue

            items.append(item)
            if len(items) >= limit:
                break

        return items

    # =========================================================
    # HELPER METHODS
    # =========================================================

    async def _fetch_entity(
        self,
        entity_id: UUID,
        entity_type: str,
    ) -> Optional[Union[KnowledgeItem, LearningAtom, PlatinumAtom]]:
        """Fetch an entity by type."""
        if entity_type == "knowledge_item":
            return await self._graph.get_knowledge_item(entity_id)
        elif entity_type == "atom":
            return await self._graph.get_atom(entity_id)
        elif entity_type == "platinum_atom":
            return await self._graph.get_platinum_atom(entity_id)
        return None

    def _matches_filters(
        self,
        entity: Union[KnowledgeItem, LearningAtom, PlatinumAtom],
        filters: Optional[Dict[str, Any]],
    ) -> bool:
        """Check if entity matches filters."""
        if not filters:
            return True

        for key, value in filters.items():
            entity_value = getattr(entity, key, None)
            if entity_value is None:
                # Check in content dict for atoms
                if hasattr(entity, "content") and isinstance(entity.content, dict):
                    entity_value = entity.content.get(key)

            if entity_value != value:
                return False

        return True
