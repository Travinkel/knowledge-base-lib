"""
Prerequisite Service - manages prerequisite relationships.

Provides operations for managing learning dependencies
and computing learning paths.
"""

from typing import Any, Dict, List, Optional, Set
from uuid import UUID

from ..models import PrerequisiteLink, LearningAtom
from ..storage import GraphStorage


class CycleDetectedError(Exception):
    """Raised when adding a prerequisite would create a cycle."""
    def __init__(self, cycle: List[UUID]):
        self.cycle = cycle
        super().__init__(f"Cycle detected: {' -> '.join(str(u) for u in cycle)}")


class PrerequisiteService:
    """
    Manages prerequisite relationships between atoms.

    Ensures the prerequisite graph remains a valid DAG
    and provides topological ordering for learning paths.
    """

    def __init__(self, graph_storage: GraphStorage):
        self._graph = graph_storage

    # =========================================================
    # PREREQUISITE MANAGEMENT
    # =========================================================

    async def add_prerequisite(
        self,
        from_atom_id: UUID,
        to_atom_id: UUID,
        strength: str = "recommended",
        rationale: str = "",
        confidence: float = 0.5,
        validate_dag: bool = True,
    ) -> PrerequisiteLink:
        """
        Add a prerequisite relationship.

        Args:
            from_atom_id: The atom that requires the prerequisite
            to_atom_id: The prerequisite atom (must be learned first)
            strength: required|recommended|optional
            rationale: Why this is a prerequisite
            confidence: Confidence in this relationship (0.0-1.0)
            validate_dag: Check for cycles before adding

        Raises:
            CycleDetectedError: If adding would create a cycle
        """
        if validate_dag:
            # Check if adding this edge would create a cycle
            would_cycle = await self._would_create_cycle(from_atom_id, to_atom_id)
            if would_cycle:
                # Find the cycle for error reporting
                cycles = await self._graph.detect_cycles()
                raise CycleDetectedError(cycles[0] if cycles else [from_atom_id, to_atom_id])

        link = PrerequisiteLink(
            from_atom_id=from_atom_id,
            to_atom_id=to_atom_id,
            strength=strength,
            rationale=rationale,
            confidence=confidence,
        )
        return await self._graph.add_prerequisite(link)

    async def remove_prerequisite(
        self,
        from_atom_id: UUID,
        to_atom_id: UUID,
    ) -> bool:
        """Remove a prerequisite relationship."""
        return await self._graph.remove_prerequisite(from_atom_id, to_atom_id)

    async def get_prerequisites(
        self,
        atom_id: UUID,
        strength: Optional[str] = None,
        include_transitive: bool = False,
    ) -> List[PrerequisiteLink]:
        """
        Get prerequisites for an atom.

        Args:
            atom_id: The atom to get prerequisites for
            strength: Filter by strength (required|recommended|optional)
            include_transitive: Include transitive prerequisites

        Returns:
            List of prerequisite links
        """
        direct = await self._graph.get_prerequisites(atom_id, strength=strength)

        if not include_transitive:
            return direct

        # Get transitive closure
        all_prereqs: Dict[UUID, PrerequisiteLink] = {}
        for link in direct:
            all_prereqs[link.to_atom_id] = link

        # BFS for transitive prerequisites
        to_visit = [link.to_atom_id for link in direct]
        visited: Set[UUID] = set()

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)

            prereqs = await self._graph.get_prerequisites(current, strength=strength)
            for link in prereqs:
                if link.to_atom_id not in all_prereqs:
                    all_prereqs[link.to_atom_id] = link
                    to_visit.append(link.to_atom_id)

        return list(all_prereqs.values())

    async def get_dependents(
        self,
        atom_id: UUID,
    ) -> List[PrerequisiteLink]:
        """Get atoms that depend on this atom as a prerequisite."""
        return await self._graph.get_dependents(atom_id)

    # =========================================================
    # LEARNING PATHS
    # =========================================================

    async def get_learning_path(
        self,
        target_atom_id: UUID,
        mastered_atom_ids: Optional[List[UUID]] = None,
    ) -> List[LearningAtom]:
        """
        Get the learning path to a target atom.

        Returns atoms in topological order (prerequisites first),
        excluding already mastered atoms.
        """
        mastered = set(mastered_atom_ids or [])

        # Get full prerequisite chain
        chain = await self._graph.get_prerequisite_chain(target_atom_id)

        # Filter out mastered atoms
        return [atom for atom in chain if atom.id not in mastered]

    async def get_next_atoms(
        self,
        mastered_atom_ids: List[UUID],
        limit: int = 5,
    ) -> List[LearningAtom]:
        """
        Get the next atoms to learn based on mastered prerequisites.

        Returns atoms whose prerequisites are all mastered.
        """
        mastered = set(mastered_atom_ids)
        candidates: List[LearningAtom] = []

        # Get all atoms
        all_atoms = await self._graph.list_atoms(limit=1000)

        for atom in all_atoms:
            if atom.id in mastered:
                continue

            # Check if all prerequisites are mastered
            prereqs = await self._graph.get_prerequisites(atom.id, strength="required")
            if all(link.to_atom_id in mastered for link in prereqs):
                candidates.append(atom)

        # Sort by number of dependents (prioritize foundational concepts)
        scored: List[tuple[LearningAtom, int]] = []
        for atom in candidates:
            dependents = await self._graph.get_dependents(atom.id)
            scored.append((atom, len(dependents)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [atom for atom, _ in scored[:limit]]

    async def topological_sort(
        self,
        atom_ids: Optional[List[UUID]] = None,
    ) -> List[UUID]:
        """Return atoms in learning order (prerequisites first)."""
        return await self._graph.topological_sort(atom_ids)

    # =========================================================
    # VALIDATION
    # =========================================================

    async def validate_dag(self) -> Dict[str, Any]:
        """Validate that the prerequisite graph is a valid DAG."""
        cycles = await self._graph.detect_cycles()
        return {
            "is_valid": len(cycles) == 0,
            "cycles": cycles,
        }

    async def _would_create_cycle(
        self,
        from_atom_id: UUID,
        to_atom_id: UUID,
    ) -> bool:
        """Check if adding an edge would create a cycle."""
        # If to_atom_id can reach from_atom_id, adding from->to creates a cycle
        path = await self._graph.find_path(to_atom_id, from_atom_id)
        return path is not None
