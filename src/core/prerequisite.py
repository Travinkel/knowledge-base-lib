"""
Prerequisite chain traversal and analysis.

Implements:
- Recursive prerequisite resolution
- Learning path generation
- Gap detection
- Topological ordering

Scientific Foundation:
- Zone of Proximal Development (Vygotsky, 1978)
- Knowledge Space Theory (Doignon & Falmagne, 1999)
"""

import logging
from dataclasses import dataclass
from enum import Enum

from .models import Atom, Concept
from .atom_store import AtomStore, InMemoryAtomStore
from .concept_graph import ConceptGraph, InMemoryConceptGraph


logger = logging.getLogger(__name__)


class MasteryStatus(str, Enum):
    """Mastery status for a concept or atom."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    MASTERED = "mastered"


@dataclass
class PrerequisiteNode:
    """A node in the prerequisite traversal tree."""
    concept: Concept
    depth: int
    mastery_status: MasteryStatus = MasteryStatus.NOT_STARTED
    children: list["PrerequisiteNode"] | None = None


@dataclass
class LearningPath:
    """A recommended learning path through prerequisites."""
    start_concept_id: str
    target_concept_id: str
    path: list[Concept]  # Ordered from start to target
    first_gap_index: int | None = None  # Index of first non-mastered concept


class PrerequisiteService:
    """
    Service for prerequisite analysis and learning path generation.
    """

    def __init__(
        self,
        atom_store: AtomStore | None = None,
        concept_graph: ConceptGraph | None = None
    ) -> None:
        self._atom_store = atom_store or InMemoryAtomStore()
        self._concept_graph = concept_graph or InMemoryConceptGraph()

    def get_all_prerequisites(
        self,
        atom_id: str,
        visited: set[str] | None = None
    ) -> list[Atom]:
        """
        Recursively get all prerequisites for an atom.

        Returns atoms in topological order (deepest dependencies first).
        Uses visited set to prevent cycles.
        """
        if visited is None:
            visited = set()

        if atom_id in visited:
            logger.warning(f"Cycle detected at atom: {atom_id}")
            return []

        visited.add(atom_id)

        atom = self._atom_store.get(atom_id)
        if not atom:
            logger.warning(f"Atom not found: {atom_id}")
            return []

        result: list[Atom] = []

        # Recurse into prerequisites first (deepest first ordering)
        for prereq_id in atom.prerequisites:
            prereq_chain = self.get_all_prerequisites(prereq_id, visited)
            result.extend(prereq_chain)

            prereq_atom = self._atom_store.get(prereq_id)
            if prereq_atom and prereq_atom not in result:
                result.append(prereq_atom)

        return result

    def get_prerequisite_tree(
        self,
        concept_id: str,
        max_depth: int = 10,
        mastery_lookup: dict[str, MasteryStatus] | None = None
    ) -> PrerequisiteNode | None:
        """
        Build a prerequisite tree for a concept.

        Returns a tree structure showing the dependency hierarchy
        with optional mastery status annotations.
        """
        concept = self._concept_graph.get_concept(concept_id)
        if not concept:
            return None

        mastery = MasteryStatus.NOT_STARTED
        if mastery_lookup:
            mastery = mastery_lookup.get(concept_id, MasteryStatus.NOT_STARTED)

        return self._build_tree(
            concept=concept,
            depth=0,
            max_depth=max_depth,
            visited=set(),
            mastery_lookup=mastery_lookup or {}
        )

    def _build_tree(
        self,
        concept: Concept,
        depth: int,
        max_depth: int,
        visited: set[str],
        mastery_lookup: dict[str, MasteryStatus]
    ) -> PrerequisiteNode:
        """Recursively build prerequisite tree."""
        mastery = mastery_lookup.get(concept.id, MasteryStatus.NOT_STARTED)
        node = PrerequisiteNode(
            concept=concept,
            depth=depth,
            mastery_status=mastery,
            children=[]
        )

        if depth >= max_depth or concept.id in visited:
            return node

        visited.add(concept.id)

        for prereq_id in concept.prerequisite_ids:
            prereq = self._concept_graph.get_concept(prereq_id)
            if prereq:
                child = self._build_tree(
                    prereq, depth + 1, max_depth, visited, mastery_lookup
                )
                node.children.append(child)

        return node

    def find_learning_path(
        self,
        target_concept_id: str,
        mastery_lookup: dict[str, MasteryStatus] | None = None
    ) -> LearningPath | None:
        """
        Find optimal learning path to a target concept.

        Returns concepts in order from foundational to target,
        identifying the first gap (non-mastered concept).
        """
        # Get all prerequisites in reverse topological order
        all_prereqs = self._concept_graph.get_all_prerequisites_recursive(target_concept_id)

        target = self._concept_graph.get_concept(target_concept_id)
        if not target:
            return None

        # Add target at the end
        path = all_prereqs + [target]

        # Find first gap
        first_gap = None
        if mastery_lookup:
            for i, concept in enumerate(path):
                status = mastery_lookup.get(concept.id, MasteryStatus.NOT_STARTED)
                if status != MasteryStatus.MASTERED:
                    first_gap = i
                    break

        # Find start (first non-mastered or beginning)
        start_id = path[0].id if path else target_concept_id

        return LearningPath(
            start_concept_id=start_id,
            target_concept_id=target_concept_id,
            path=path,
            first_gap_index=first_gap
        )

    def detect_missing_prerequisites(
        self,
        concept_id: str
    ) -> list[str]:
        """
        Detect concepts that are likely missing prerequisites.

        A concept is flagged if it has no prerequisites but is at
        a non-beginner difficulty level (> 0.3).
        """
        concept = self._concept_graph.get_concept(concept_id)
        if not concept:
            return []

        issues: list[str] = []

        if not concept.prerequisite_ids and concept.difficulty > 0.3:
            issues.append(
                f"Concept '{concept.title}' has difficulty {concept.difficulty} "
                "but no prerequisites defined"
            )

        return issues
