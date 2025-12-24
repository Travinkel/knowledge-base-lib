@Domain-KB @Vector @Embeddings @Semantic @Priority-Regression
Feature: Vector Search and Semantic Similarity
  As the Knowledge Base Service
  I want to store and query vector embeddings for semantic search
  So that learners can find conceptually related content beyond keyword matching.

  Background:
    Given the PostgreSQL database has pgvector extension enabled
    And embedding models are configured:
      | Model              | Dimensions | Use_Case                    |
      | text-embedding-3   | 1536       | General concept embeddings  |
      | code-embedding     | 768        | Code snippet similarity     |
      | multimodal-embed   | 1024       | Text + diagram alignment    |
    And the vector index uses HNSW algorithm with:
      | Parameter    | Value |
      | m            | 16    |
      | ef_construct | 64    |
      | ef_search    | 40    |

  # ─────────────────────────────────────────────────────────────────────────
  # Embedding Storage
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @Embedding @Storage
  Scenario: Storing embeddings for concept nodes
    Given a concept "Quicksort Algorithm" with content:
      | Field         | Content                                    |
      | Title         | Quicksort Algorithm                        |
      | Summary       | Divide-and-conquer sorting algorithm...    |
      | Key_Points    | Pivot selection, Partitioning, Recursion   |
    When embedding generation executes
    Then vectors should be stored:
      | Vector_Type   | Dimensions | Source              |
      | Title_Embed   | 1536       | Title text          |
      | Summary_Embed | 1536       | Full summary        |
      | Combined_Embed| 1536       | Weighted combination|
    And metadata should include generation timestamp and model version

  @Positive @Embedding @Atoms
  Scenario: Storing embeddings for learning atoms
    Given an MCQ atom with content:
      | Field         | Content                                    |
      | Stem          | What is the average time complexity...     |
      | Options       | [O(n), O(n log n), O(n²), O(log n)]       |
      | Explanation   | Quicksort uses divide-and-conquer...       |
    When atom embedding generation executes
    Then vectors should be stored:
      | Vector_Type       | Source                | Weight |
      | Stem_Embed        | Question stem         | 0.5    |
      | Explanation_Embed | Full explanation      | 0.3    |
      | Combined_Embed    | Weighted combination  | 1.0    |
    And atom should be linkable via semantic similarity to concepts

  @Positive @Embedding @CodeBlocks
  Scenario: Storing code-specific embeddings
    Given a code atom with:
      | Language | Code_Content                              |
      | Python   | def quicksort(arr): ...                   |
    When code embedding generation executes
    Then it should:
      | Action                | Details                        |
      | Parse_AST             | Extract structure              |
      | Generate_Doc_Embed    | Natural language description   |
      | Generate_Code_Embed   | Code-specific embedding        |
      | Store_Both            | Dual embedding for hybrid search|
    And code can be found via both "sort algorithm" and similar code patterns

  # ─────────────────────────────────────────────────────────────────────────
  # k-NN Similarity Queries
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Search @kNN
  Scenario: Finding k nearest neighbor concepts
    Given a query: "How do hash tables handle collisions?"
    When k-NN search executes with k=5
    Then results should include:
      | Rank | Concept              | Similarity | Domain          |
      | 1    | Hash_Collision       | 0.94       | Data_Structures |
      | 2    | Open_Addressing      | 0.89       | Data_Structures |
      | 3    | Chaining             | 0.87       | Data_Structures |
      | 4    | Hash_Functions       | 0.82       | Data_Structures |
      | 5    | Load_Factor          | 0.78       | Data_Structures |
    And query latency should be < 50ms
    And results should be ranked by cosine similarity

  @Positive @Search @SemanticExpansion
  Scenario: Expanding search with semantic synonyms
    Given a query: "memory leak"
    When semantic search with expansion executes
    Then it should:
      | Expansion_Type    | Terms_Added                          |
      | Synonyms          | memory_leak, resource_leak           |
      | Related_Concepts  | garbage_collection, heap_allocation  |
      | Technical_Variants| memory_corruption, dangling_pointer  |
    And return results matching any expansion term
    And rank by combined relevance score

  @Positive @Search @CrossModal
  Scenario: Cross-modal search (text to diagram)
    Given a text query: "binary tree traversal order"
    And diagrams have multimodal embeddings
    When cross-modal search executes
    Then it should return:
      | Result_Type | Content                    | Similarity |
      | Diagram     | Inorder_Traversal_Viz      | 0.88       |
      | Diagram     | Preorder_Traversal_Viz     | 0.85       |
      | Diagram     | Tree_Structure_Diagram     | 0.79       |
    And text queries can surface visual learning materials

  # ─────────────────────────────────────────────────────────────────────────
  # Hybrid Search (Graph + Vector)
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Hybrid @GraphVector
  Scenario: Combining graph traversal with vector similarity
    Given learner is studying "Recursion"
    And wants to explore related concepts
    When hybrid search executes:
      | Search_Type   | Weight | Query                        |
      | Graph_Prereqs | 0.4    | Concepts linked to Recursion |
      | Vector_kNN    | 0.4    | Semantically similar         |
      | Keyword       | 0.2    | Text match "recursive"       |
    Then results should be ranked by combined score:
      | Concept           | Graph_Score | Vector_Score | Combined |
      | Tail_Recursion    | 0.9         | 0.92         | 0.91     |
      | Stack_Frames      | 0.8         | 0.75         | 0.77     |
      | Divide_Conquer    | 0.3         | 0.88         | 0.59     |
      | Iteration         | 0.7         | 0.45         | 0.55     |
    And hybrid approach surfaces both structural and semantic relations

  @Positive @Hybrid @FilteredSearch
  Scenario: Filtering vector search by graph properties
    Given a semantic query: "algorithm efficiency"
    And filter constraints:
      | Filter            | Value                    |
      | Domain            | Computer_Science         |
      | Difficulty_Range  | 0.4 - 0.7                |
      | Has_Atoms         | true                     |
    When filtered vector search executes
    Then it should:
      | Step              | Action                              |
      | Pre-filter        | Apply graph constraints first       |
      | Vector_Search     | k-NN within filtered set            |
      | Post-rank         | Combine similarity + filter scores  |
    And only return concepts matching all filters
    And maintain search performance via hybrid indexing

  # ─────────────────────────────────────────────────────────────────────────
  # Similarity Clustering
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Clustering @AutoGrouping
  Scenario: Auto-clustering concepts by semantic similarity
    Given 100 ungrouped concepts in "Algorithms" domain
    When semantic clustering executes with target_clusters=8
    Then it should produce:
      | Cluster_ID | Centroid_Concept    | Members | Cohesion |
      | 1          | Sorting_Overview    | 12      | 0.85     |
      | 2          | Graph_Traversal     | 15      | 0.82     |
      | 3          | Dynamic_Programming | 10      | 0.88     |
      | 4          | String_Algorithms   | 8       | 0.79     |
      | 5          | Search_Algorithms   | 14      | 0.84     |
      | 6          | Greedy_Algorithms   | 11      | 0.81     |
      | 7          | Divide_Conquer      | 9       | 0.86     |
      | 8          | Computational_Geom  | 6       | 0.77     |
    And clusters should inform learning module boundaries

  @Positive @Clustering @OutlierDetection
  Scenario: Detecting semantic outliers for review
    Given concept embeddings for "Data Structures" domain
    When outlier detection analyzes cluster distances
    Then it should flag:
      | Concept           | Distance_From_Centroid | Status    |
      | Arrays            | 0.12                   | Normal    |
      | Linked_Lists      | 0.15                   | Normal    |
      | Blockchain        | 0.78                   | OUTLIER   |
      | B_Trees           | 0.22                   | Normal    |
    And outliers should be reviewed for domain misclassification
    And recommendation: "Move Blockchain to Cryptography domain?"

  # ─────────────────────────────────────────────────────────────────────────
  # Embedding Quality & Maintenance
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Quality @DriftDetection
  Scenario: Detecting embedding drift after content updates
    Given concept "Machine Learning" was updated with new content
    And original embedding was generated 30 days ago
    When embedding drift check runs
    Then it should compute:
      | Metric              | Value  | Threshold | Action        |
      | Content_Hash_Change | Yes    | -         | Trigger check |
      | Embedding_Distance  | 0.35   | 0.20      | Re-embed      |
      | Stale_Age_Days      | 30     | 90        | OK            |
    And schedule re-embedding due to content change
    And log: "ML concept embedding outdated by content update"

  @Positive @Quality @Deduplication
  Scenario: Detecting near-duplicate concepts via embeddings
    Given two concepts:
      | Concept_A          | Concept_B             |
      | Array_Lists        | Dynamic_Arrays        |
    When similarity analysis runs
    Then if cosine_similarity > 0.95:
      | Action              | Details                        |
      | Flag_Potential_Dupe | Similarity 0.97                |
      | Compare_Properties  | Check if truly duplicate       |
      | Suggest_Merge       | "Consider merging concepts"    |
    And human review should confirm merge decision

  @Positive @Quality @BatchReembedding
  Scenario: Batch re-embedding after model upgrade
    Given 5000 concepts have embeddings from model v1
    And model v2 is now available with better performance
    When batch re-embedding job executes
    Then it should:
      | Phase               | Action                         |
      | Backup              | Store v1 embeddings            |
      | Generate            | Create v2 embeddings in batch  |
      | Validate            | Compare search quality metrics |
      | Swap                | Atomic switch to v2 index      |
    And rollback should be available if v2 quality degrades

  # ─────────────────────────────────────────────────────────────────────────
  # Advanced Vector Operations
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Advanced @AnalogicalSearch
  Scenario: Analogical reasoning via vector arithmetic
    Given the analogy: "King - Man + Woman = ?"
    When vector arithmetic search executes:
      | Operation         | Vectors                         |
      | Start             | embed("King")                   |
      | Subtract          | embed("Man")                    |
      | Add               | embed("Woman")                  |
      | Search            | Nearest to result vector        |
    Then result should be: "Queen" (or semantically equivalent)
    And this enables: "Recursion - Programming + Math = Mathematical_Induction"

  @Positive @Advanced @ConceptInterpolation
  Scenario: Finding concepts between two anchor points
    Given anchor concepts:
      | Concept_A         | Concept_B           |
      | Sorting_Algorithms| Machine_Learning    |
    When interpolation search finds concepts along vector path
    Then it should return:
      | Position | Concept                    | Distance_From_A |
      | 0.0      | Sorting_Algorithms         | 0.00            |
      | 0.25     | Algorithm_Complexity       | 0.25            |
      | 0.50     | Optimization_Algorithms    | 0.50            |
      | 0.75     | Gradient_Descent           | 0.75            |
      | 1.0      | Machine_Learning           | 1.00            |
    And this suggests learning path from sorting to ML

  @Positive @Advanced @MultiVector
  Scenario: Multi-vector concept representation
    Given concept "Recursion" has multiple facets:
      | Facet             | Description                    |
      | Definition        | What recursion is              |
      | Implementation    | How to code recursion          |
      | Applications      | Where recursion is used        |
      | Pitfalls          | Common recursion mistakes      |
    When multi-vector representation is stored
    Then searches can target specific facets:
      | Query                          | Best_Facet_Match |
      | "recursive function definition"| Definition       |
      | "stack overflow recursion"     | Pitfalls         |
      | "tree traversal recursive"     | Applications     |
    And facet-specific retrieval improves relevance

  # ─────────────────────────────────────────────────────────────────────────
  # Performance Optimization
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Performance @IndexTuning
  Scenario: Tuning HNSW index for query patterns
    Given query pattern analysis shows:
      | Pattern           | Frequency | Avg_k |
      | Exact_kNN         | 60%       | 5     |
      | Filtered_kNN      | 30%       | 10    |
      | Range_Search      | 10%       | 20    |
    When index tuning recommends:
      | Parameter    | Current | Recommended | Reason                |
      | ef_search    | 40      | 64          | Higher recall needed  |
      | m            | 16      | 24          | Larger graph degree   |
    Then applying recommendations should:
      | Metric           | Before | After  |
      | Recall@10        | 0.92   | 0.97   |
      | Query_Latency_ms | 45     | 52     |
    And accept slight latency increase for better recall

  @Positive @Performance @Quantization
  Scenario: Applying vector quantization for scale
    Given 1 million concept embeddings consume 6GB storage
    When scalar quantization (int8) is applied
    Then storage should reduce:
      | Metric           | Before    | After     | Reduction |
      | Storage_GB       | 6.0       | 1.5       | 75%       |
      | Search_Latency   | 45ms      | 38ms      | 15% faster|
      | Recall@10        | 0.97      | 0.95      | -2%       |
    And minor recall loss is acceptable for 4x storage savings

