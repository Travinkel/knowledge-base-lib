@KnowledgeBase @Search @Vector @Priority-High
Feature: Semantic Vector Search
  As a RAG pipeline
  I want to search atoms by semantic similarity
  So that I can retrieve relevant context

  Background:
    Given the Knowledge Base service is initialized
    And the vector storage backend is connected
    And embeddings are generated using "text-embedding-ada-002"

  # =========================================================
  # EMBEDDING STORAGE
  # =========================================================

  @Smoke @Embedding
  Scenario: Store embedding for a learning atom
    Given a learning atom with content about "OSPF routing protocol"
    When I store an embedding for the atom
    Then the embedding should be persisted
    And the embedding dimension should match the model (1536)

  @Regression @Embedding
  Scenario: Batch store embeddings
    Given 10 learning atoms without embeddings
    When I batch store embeddings for all atoms
    Then all 10 embeddings should be persisted
    And each should reference the correct atom ID

  @Regression @Embedding
  Scenario: Update existing embedding
    Given an atom with an existing embedding
    When I update the embedding with a new vector
    Then the old embedding should be replaced
    And the new embedding should be queryable

  # =========================================================
  # SIMILARITY SEARCH
  # =========================================================

  @Smoke @Search
  Scenario: Search returns semantically similar atoms
    Given 100 atoms with embeddings covering networking topics
    When I search for "how do networks route packets"
    Then results should include atoms about:
      | Concept           | Min Similarity |
      | Routing Tables    | 0.7            |
      | IP Forwarding     | 0.6            |
      | Packet Switching  | 0.6            |
    And results should be ordered by similarity descending

  @Smoke @Search
  Scenario: Search with minimum similarity threshold
    Given atoms with varying relevance to "subnetting"
    When I search for "subnetting" with min_similarity 0.8
    Then all results should have similarity >= 0.8
    And irrelevant atoms should be excluded

  @Regression @Search
  Scenario: Search with top_k limit
    Given 100 atoms about various topics
    When I search for "networking basics" with top_k=5
    Then I should get exactly 5 results
    And they should be the 5 most similar atoms

  @Regression @Search
  Scenario: Search filtered by entity type
    Given atoms and knowledge items about "machine learning"
    When I search for "neural networks" with entity_type="atom"
    Then results should only include learning atoms
    And knowledge items should be excluded

  # =========================================================
  # HYBRID SEARCH
  # =========================================================

  @Smoke @Hybrid
  Scenario: Hybrid search combines vector and graph
    Given atoms about "recursion" with prerequisites
    When I perform hybrid search for "recursive algorithms"
    Then results should include direct semantic matches
    And results should also include prerequisite atoms
    And each result should indicate its source (semantic/graph)

  @Regression @Hybrid
  Scenario: Hybrid search with alpha weighting
    Given atoms with varying semantic and graph relevance
    When I search with alpha=0.3 (favor graph)
    Then graph-connected atoms should rank higher
    And pure semantic matches should rank lower

  @Regression @Hybrid
  Scenario: Hybrid search includes analogies
    Given atom "Binary Search" with analogy to "Number Guessing"
    When I search for "efficient search algorithms" with include_analogies=true
    Then results should include "Binary Search"
    And results should also include "Number Guessing" via analogy

  # =========================================================
  # SPECIALIZED SEARCHES
  # =========================================================

  @Smoke @Specialized
  Scenario: Find atoms for a concept with ICAP filter
    Given atoms about "loops" at various ICAP levels:
      | atom               | icap_level   |
      | Loop Concept       | passive      |
      | Loop Exercise      | active       |
      | Build Loop Pattern | constructive |
    When I search for "programming loops" with icap_level="constructive"
    Then results should only include "Build Loop Pattern"

  @Regression @Specialized
  Scenario: Find atoms within difficulty range
    Given atoms about "algorithms" with difficulties:
      | atom              | difficulty |
      | Basic Sorting     | 0.3        |
      | Binary Search     | 0.5        |
      | Dynamic Programming| 0.9       |
    When I search with difficulty_range=(0.4, 0.6)
    Then results should only include "Binary Search"

  @Smoke @Specialized
  Scenario: Find evidence for a claim
    Given knowledge items from various study types:
      | item                  | study_type    | confidence |
      | Spacing Effect Meta   | meta_analysis | 0.95       |
      | Spacing Case Study    | case_study    | 0.70       |
    When I search for evidence about "spaced repetition" with min_confidence=0.8
    Then results should include "Spacing Effect Meta"
    And "Spacing Case Study" should be excluded

  # =========================================================
  # INDEX MANAGEMENT
  # =========================================================

  @Smoke @Index
  Scenario: Create HNSW index for fast search
    Given embeddings for 10000 atoms
    When I create an HNSW index with m=16, ef_construct=64
    Then the index should be created successfully
    And search queries should complete in < 100ms

  @Regression @Index
  Scenario: Rebuild index after bulk updates
    Given an existing index with 5000 embeddings
    And 1000 new embeddings added
    When I rebuild the index
    Then all 6000 embeddings should be indexed
    And search should return newly added atoms

  @Regression @Index
  Scenario: Get index statistics
    Given an index with 10000 embeddings
    When I request index stats
    Then stats should include:
      | metric          | type   |
      | total_vectors   | int    |
      | dimension       | int    |
      | index_type      | string |
      | avg_query_time  | float  |

  # =========================================================
  # PERFORMANCE
  # =========================================================

  @Performance @NonFunctional
  Scenario: Vector search latency under load
    Given 100000 atoms with embeddings
    And an HNSW index configured optimally
    When I perform 100 concurrent search queries
    Then p95 latency should be < 100ms
    And all queries should return results

  @Performance @NonFunctional
  Scenario: Embedding storage throughput
    Given 1000 atoms to embed
    When I batch store all embeddings
    Then throughput should be >= 100 embeddings/second
    And no embeddings should be lost
