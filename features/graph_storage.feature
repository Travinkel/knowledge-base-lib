@KnowledgeBase @Graph @Priority-High @Regression
@Node-KnowledgeBase @Edge-KB-DB @DistributedSystems
Feature: Knowledge Graph Storage and Retrieval
  """
  The Knowledge Base provides persistent storage for the knowledge graph,
  combining relational data (PostgreSQL) with vector embeddings (pgvector)
  for hybrid retrieval.

  Architecture:
  - PostgreSQL: Entities, relationships, metadata (ACID transactions)
  - pgvector: High-dimensional embeddings (semantic similarity)
  - Graph queries: Traversal via recursive CTEs or adjacency patterns

  Scientific Foundation:
  - Bordes et al. (2013). Knowledge Graph Embeddings
  - Hamilton et al. (2017). GraphSAGE - Inductive representation learning
  - Lewis et al. (2020). RAG - Retrieval-Augmented Generation

  Key Operations:
  - Store/retrieve learning atoms and concepts
  - Query prerequisite chains (transitive closure)
  - Semantic similarity search for related content
  - Subgraph extraction for context building
  """

  Background:
    Given the PostgreSQL database is running
    And pgvector extension is enabled
    And the knowledge base service is initialized

  # ============================================================================
  # CRUD OPERATIONS
  # ============================================================================

  @Smoke @CRUD @Entity-LearningAtom
  Scenario: Store and retrieve a learning atom
    Given a learning atom with:
      | field          | value                              |
      | id             | atom-uuid-123                      |
      | atom_type      | mcq                                |
      | knowledge_type | conceptual                         |
      | icap_level     | constructive                       |
      | content        | {"question": "...", "options": []} |
      | concept_id     | concept-subnet-001                 |
    When the atom is stored
    Then the atom can be retrieved by ID
    And the atom can be retrieved by concept_id
    And the atom can be retrieved by atom_type

  @Smoke @CRUD @Entity-Concept
  Scenario: Store concept with hierarchy
    Given a concept hierarchy:
      | id              | name           | parent_cluster_id |
      | concept-ipv4-001| IPv4 Addressing| cluster-networking-001 |
    When the concept is stored
    Then it is linked to its cluster
    And it can be traversed from ConceptArea down

  @Regression @CRUD @Relationships
  Scenario: Store relationships between entities
    Given atoms exist:
      | atom_id     | concept_id       |
      | atom-001    | concept-binary   |
      | atom-002    | concept-subnet   |
    When a prerequisite relationship is created:
      | from_concept    | to_concept      | type             |
      | concept-binary  | concept-subnet  | hard_prerequisite|
    Then the relationship is persisted
    And bidirectional traversal is possible

  # ============================================================================
  # GRAPH TRAVERSAL
  # ============================================================================

  @Smoke @GraphTraversal @Prerequisites
  Scenario: Query all prerequisites for a concept (transitive closure)
    """
    Uses recursive CTE to find all transitive prerequisites.
    Essential for prerequisite gating before presenting advanced content.
    """
    Given the concept hierarchy:
      """
      binary_numbers → IP_addressing → subnetting → VLSM → route_summarization
      """
    When querying all prerequisites of "route_summarization"
    Then the result includes:
      | concept          | depth |
      | VLSM             | 1     |
      | subnetting       | 2     |
      | IP_addressing    | 3     |
      | binary_numbers   | 4     |
    And the query uses recursive CTE for efficiency

  @Regression @GraphTraversal @Descendants
  Scenario: Query all concepts that depend on a concept
    """
    Find all downstream concepts - useful for impact analysis
    when a foundational concept is not mastered.
    """
    Given "binary_numbers" is a foundational concept
    When querying all descendants of "binary_numbers"
    Then the result includes all concepts that transitively require it
    And the result is ordered by dependency depth

  @Regression @GraphTraversal @ShortestPath
  Scenario: Find shortest learning path between concepts
    """
    Dijkstra's algorithm finds the optimal path through the knowledge
    graph, weighted by estimated learning time.
    """
    Given learner wants to learn "route_summarization"
    And learner has mastered "binary_numbers"
    When the shortest learning path is computed
    Then the path is:
      | step | concept        | estimated_time |
      | 1    | IP_addressing  | 45 min         |
      | 2    | subnetting     | 60 min         |
      | 3    | VLSM           | 45 min         |
      | 4    | route_summarization | 30 min    |
    And total estimated time is 180 minutes

  # ============================================================================
  # SEMANTIC SEARCH (pgvector)
  # ============================================================================

  @Smoke @SemanticSearch @pgvector
  Scenario: Find similar atoms by semantic similarity
    """
    Uses cosine similarity in pgvector for efficient nearest neighbor search.
    Enables "find related content" and context augmentation.
    """
    Given atoms are embedded in vector space
    When semantic search is executed with query "how to calculate subnet masks"
    Then results are ranked by cosine similarity:
      | atom_id     | similarity | title                        |
      | atom-sub-001| 0.92       | Subnet Mask Calculation      |
      | atom-sub-002| 0.87       | CIDR Notation Conversion     |
      | atom-sub-003| 0.81       | Network Address Determination|
    And search uses pgvector index (IVFFlat or HNSW)

  @Regression @SemanticSearch @Filtering
  Scenario: Filtered semantic search by metadata
    """
    Combine vector similarity with metadata filters for precise retrieval.
    Example: "Find conceptual atoms about subnetting at constructive ICAP level"
    """
    Given the semantic query "subnet calculation methods"
    And filters:
      | field          | value        |
      | knowledge_type | procedural   |
      | icap_level     | constructive |
      | difficulty     | < 0.7        |
    When filtered semantic search is executed
    Then results match both semantic similarity AND filters
    And results are sorted by similarity within filter constraints

  @Regression @SemanticSearch @HybridRetrieval
  Scenario: Hybrid retrieval combining keyword and semantic search
    """
    RAG pattern: combine BM25 keyword matching with vector similarity
    for optimal retrieval (Lewis et al., 2020).
    """
    Given query "255.255.255.0 subnet mask class C"
    When hybrid retrieval is executed
    Then results combine:
      | retrieval_type | weight | purpose                    |
      | keyword_bm25   | 0.4    | exact term matching        |
      | semantic_vector| 0.6    | conceptual understanding   |
    And the final ranking uses reciprocal rank fusion

  # ============================================================================
  # SUBGRAPH EXTRACTION
  # ============================================================================

  @Smoke @Subgraph @ContextBuilding
  Scenario: Extract subgraph for context augmentation
    """
    For LLM prompts, extract a relevant subgraph containing:
    - The target concept
    - Prerequisites (bounded depth)
    - Related atoms
    - Analogies from other domains
    """
    Given target concept "subnetting"
    When subgraph extraction is executed with:
      | parameter          | value |
      | prerequisite_depth | 2     |
      | atom_limit         | 10    |
      | include_analogies  | true  |
    Then the subgraph contains:
      | element_type   | count |
      | concepts       | 5     |
      | atoms          | 10    |
      | relationships  | 8     |
    And the subgraph is serializable to JSON for LLM context

  @Regression @Subgraph @LearnerSpecific
  Scenario: Extract learner-specific subgraph based on mastery
    """
    The subgraph should be personalized: include only concepts
    the learner hasn't mastered, weighted by urgency.
    """
    Given learner "L1" has mastery:
      | concept        | mastery_score |
      | binary_numbers | 0.95          |
      | IP_addressing  | 0.72          |
      | subnetting     | 0.45          |
    When learner-specific subgraph is extracted for "VLSM"
    Then the subgraph emphasizes:
      | concept        | included | reason              |
      | binary_numbers | no       | already mastered    |
      | IP_addressing  | partial  | review recommended  |
      | subnetting     | yes      | prerequisite gap    |
      | VLSM           | yes      | target concept      |

  # ============================================================================
  # CONSISTENCY AND INTEGRITY
  # ============================================================================

  @Smoke @Integrity @Transactions
  Scenario: Atomic operations maintain consistency
    """
    Multi-step operations (e.g., store atom + create relationships)
    must be atomic to maintain referential integrity.
    """
    Given a new atom with prerequisite relationships
    When the atom is stored in a transaction
    And the transaction includes relationship creation
    Then either all operations succeed or all are rolled back
    And no orphaned relationships exist

  @Regression @Integrity @CycleDetection
  Scenario: Detect and prevent prerequisite cycles
    """
    Prerequisite relationships must form a DAG (directed acyclic graph).
    Cycles would make learning impossible.
    """
    Given existing prerequisite: "A → B → C"
    When attempting to create relationship "C → A"
    Then the operation is REJECTED
    And the error message indicates cycle detected
    And the graph remains cycle-free

  @Regression @Integrity @OrphanCleanup
  Scenario: Garbage collect orphaned entities
    """
    Atoms without concepts or concepts without areas should be
    flagged for review or cleanup.
    """
    When orphan detection is executed
    Then orphaned entities are identified:
      | entity_type | orphan_condition                |
      | atom        | concept_id is NULL or invalid   |
      | concept     | cluster_id is NULL or invalid   |
      | relationship| either end doesn't exist        |
    And orphans are reported for manual review

  # ============================================================================
  # PERFORMANCE
  # ============================================================================

  @Performance @Index @Query
  Scenario: Query performance meets SLA
    """
    Critical queries must complete within acceptable latency
    for real-time adaptive learning.
    """
    Given a knowledge base with:
      | entity_type | count    |
      | concepts    | 10,000   |
      | atoms       | 100,000  |
      | embeddings  | 100,000  |
    When benchmark queries are executed:
      | query_type            | target_latency |
      | get_atom_by_id        | < 5ms          |
      | semantic_search_top_10| < 50ms         |
      | prerequisite_chain    | < 20ms         |
      | subgraph_extraction   | < 100ms        |
    Then all queries meet their latency targets
    And indexes are properly utilized (no seq scans)

  @Performance @Embedding @Batch
  Scenario: Batch embedding ingestion
    """
    Bulk loading of embeddings should be efficient for
    initial data load and reindexing.
    """
    Given 10,000 atoms need embedding
    When batch embedding ingestion is executed
    Then throughput is >= 500 atoms/second
    And pgvector index is updated incrementally
    And memory usage stays within bounds
