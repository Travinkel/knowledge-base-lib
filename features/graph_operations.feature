@Domain-KB @Graph @Neo4j @CRUD @Priority-Regression
Feature: Knowledge Graph Operations and Traversal
  As the Knowledge Base Service
  I want to perform CRUD operations on concept nodes and atom edges
  So that the learning ontology can be queried and updated for adaptive navigation.

  Background:
    Given the Neo4j graph database is initialized
    And the PostgreSQL master ledger is synchronized
    And the ontology schema defines:
      | Node_Type       | Properties                                |
      | Concept         | id, title, domain, difficulty, created_at |
      | Atom            | id, type, icap_mode, irt_params, content  |
      | Author          | id, name, expertise, bible_ref            |
      | Domain          | id, name, parent_domain, depth            |
    And the edge schema defines:
      | Edge_Type       | From          | To            | Properties           |
      | PREREQUISITE    | Concept       | Concept       | strength, inferred   |
      | TEACHES         | Atom          | Concept       | coverage, mode       |
      | AUTHORED_BY     | Atom          | Author        | confidence           |
      | BELONGS_TO      | Concept       | Domain        | primary              |
      | ISOMORPHIC_TO   | Concept       | Concept       | mapping_score        |

  # ─────────────────────────────────────────────────────────────────────────
  # Node CRUD Operations
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Smoke @CRUD @Create
  Scenario: Creating a new concept node
    Given a concept definition:
      | Field         | Value                          |
      | title         | Binary Search Algorithm        |
      | domain        | Computer_Science.Algorithms    |
      | difficulty    | 0.65                           |
      | prerequisites | [Array_Indexing, Comparison]   |
    When the CREATE operation executes
    Then a new Concept node should exist with:
      | Property      | Value                          |
      | id            | UUID (auto-generated)          |
      | title         | Binary Search Algorithm        |
      | created_at    | Current timestamp              |
    And PREREQUISITE edges should connect to [Array_Indexing, Comparison]
    And the node should be indexed for full-text search

  @Positive @CRUD @Read
  Scenario: Reading concept with full relationship expansion
    Given concept "Recursion" exists with:
      | Relationship      | Connected_Nodes                    |
      | PREREQUISITE      | [Stack_Memory, Function_Calls]     |
      | TEACHES (inverse) | [15 atoms across ICAP modes]       |
      | ISOMORPHIC_TO     | [Mathematical_Induction]           |
    When READ operation requests full expansion
    Then response should include:
      | Component         | Content                            |
      | Core_Properties   | id, title, domain, difficulty      |
      | Prerequisites     | 2 concepts with strength scores    |
      | Teaching_Atoms    | 15 atoms grouped by ICAP mode      |
      | Isomorphisms      | 1 cross-domain mapping             |
    And response time should be < 100ms for cached nodes

  @Positive @CRUD @Update
  Scenario: Updating concept properties and relationships
    Given concept "Sorting Algorithms" has difficulty 0.50
    When UPDATE operation modifies:
      | Field           | Old_Value | New_Value |
      | difficulty      | 0.50      | 0.62      |
      | new_prereq      | -         | Time_Complexity |
    Then the node should reflect new difficulty
    And a new PREREQUISITE edge should exist to "Time_Complexity"
    And the changelog should record:
      | Event           | Details                        |
      | PROPERTY_UPDATE | difficulty: 0.50 → 0.62        |
      | EDGE_CREATED    | PREREQUISITE → Time_Complexity |

  @Positive @CRUD @Delete
  Scenario: Safe deletion with orphan prevention
    Given concept "Deprecated_Concept" has:
      | Relationship      | Count |
      | PREREQUISITE (in) | 0     |
      | PREREQUISITE (out)| 2     |
      | TEACHES (in)      | 3     |
    When DELETE operation is requested
    Then system should check orphan impact:
      | Check               | Result                         |
      | Blocking_Prereqs    | 0 (safe to delete)             |
      | Orphaned_Atoms      | 3 atoms need reassignment      |
    And deletion should be blocked until atoms are reassigned
    And warning: "3 atoms reference this concept"

  # ─────────────────────────────────────────────────────────────────────────
  # Prerequisite Graph Operations
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Graph @Prerequisites @Traversal
  Scenario: Traversing prerequisite chain to root concepts
    Given concept "Neural Networks" with deep prerequisite chain
    When prerequisite traversal executes with max_depth=10
    Then it should return ordered path:
      | Depth | Concept                    | Mastery_Status |
      | 0     | Neural_Networks            | Current        |
      | 1     | Gradient_Descent           | Mastered       |
      | 2     | Partial_Derivatives        | Mastered       |
      | 3     | Multivariable_Calculus     | In_Progress    |
      | 4     | Single_Variable_Calculus   | Mastered       |
      | 5     | Algebra_Fundamentals       | Mastered       |
    And path should identify first non-mastered concept (depth 3)
    And learning recommendation should start at depth 3

  @Positive @Graph @Prerequisites @CycleDetection
  Scenario: Detecting and preventing prerequisite cycles
    Given concept "A" has prerequisite "B"
    And concept "B" has prerequisite "C"
    When attempting to add prerequisite "A" to concept "C"
    Then cycle detection should identify: C → A → B → C
    And the operation should be REJECTED
    And error: "Prerequisite cycle detected: C → A → B → C"

  @Positive @Graph @Prerequisites @Inference
  Scenario: Inferring prerequisite relationships from content similarity
    Given concepts without explicit prerequisites:
      | Concept             | Content_Embedding              |
      | Heap_Sort           | [0.82, 0.45, ...]              |
      | Binary_Heap         | [0.85, 0.43, ...]              |
      | Tree_Data_Structure | [0.78, 0.50, ...]              |
    When prerequisite inference analyzes embeddings
    Then it should suggest:
      | Suggested_Prereq    | For_Concept  | Confidence | Reasoning          |
      | Binary_Heap         | Heap_Sort    | 0.92       | High similarity    |
      | Tree_Data_Structure | Binary_Heap  | 0.85       | Structural parent  |
    And suggestions should be flagged as "inferred" (not verified)

  # ─────────────────────────────────────────────────────────────────────────
  # Cross-Domain Isomorphism
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Graph @Isomorphism @AToM
  Scenario: Creating cross-domain isomorphic mappings
    Given concepts in different domains:
      | Concept           | Domain       |
      | Water_Flow        | Physics      |
      | Electrical_Current| Electronics  |
      | Heat_Transfer     | Thermodynamics|
    And structural analysis reveals shared relational pattern:
      | Shared_Structure    | Mapping                           |
      | Gradient_Driver     | Pressure, Voltage, Temperature    |
      | Flow_Rate           | Volume/s, Amperes, Watts          |
      | Resistance          | Viscosity, Ohms, R-value          |
    When ISOMORPHIC_TO edges are created
    Then edges should include mapping details:
      | From              | To                 | Mapping_Score |
      | Water_Flow        | Electrical_Current | 0.94          |
      | Electrical_Current| Heat_Transfer      | 0.89          |
      | Water_Flow        | Heat_Transfer      | 0.87          |
    And these mappings enable AToM transfer learning

  @Positive @Graph @Isomorphism @TransferPath
  Scenario: Finding transfer learning paths via isomorphisms
    Given learner has mastered "Gradient_Descent" in ML domain
    And target concept "Simulated_Annealing" in Optimization domain
    When transfer path finder executes
    Then it should identify:
      | Step | Concept               | Domain       | Isomorphism_To       |
      | 1    | Gradient_Descent      | ML           | -                    |
      | 2    | Hill_Climbing         | Optimization | Gradient_Descent     |
      | 3    | Simulated_Annealing   | Optimization | Hill_Climbing (ext)  |
    And suggest learning path leveraging transfer

  # ─────────────────────────────────────────────────────────────────────────
  # Graph Analytics
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Analytics @Centrality
  Scenario: Computing concept centrality for curriculum design
    Given the knowledge graph has 500 concepts
    When centrality analysis executes
    Then it should compute:
      | Concept             | PageRank | Betweenness | Interpretation        |
      | Functions           | 0.089    | 0.145       | Critical hub          |
      | Variables           | 0.075    | 0.120       | Foundational gateway  |
      | Recursion           | 0.062    | 0.098       | Integration point     |
      | Monads              | 0.012    | 0.008       | Specialized leaf      |
    And high-centrality concepts should be prioritized in curriculum
    And "bottleneck" concepts (high betweenness) flagged for extra atoms

  @Positive @Analytics @Clustering
  Scenario: Identifying concept clusters for module design
    Given the knowledge graph spans multiple domains
    When community detection algorithm runs
    Then it should identify clusters:
      | Cluster_ID | Concepts                                    | Suggested_Module      |
      | 1          | [Arrays, Lists, Trees, Graphs]              | Data_Structures       |
      | 2          | [Sorting, Searching, Dynamic_Programming]   | Algorithms            |
      | 3          | [Classes, Inheritance, Polymorphism]        | OOP_Fundamentals      |
      | 4          | [HTTP, REST, WebSockets]                    | Web_Protocols         |
    And cluster boundaries should inform learning path transitions

  @Positive @Analytics @GapDetection
  Scenario: Detecting structural gaps in knowledge graph
    Given domain "Machine Learning" has concepts:
      | Concept               | Has_Prereqs | Is_Prereq_Of |
      | Linear_Regression     | Yes         | Yes          |
      | Logistic_Regression   | Yes         | Yes          |
      | Decision_Trees        | NO          | Yes          |
      | Random_Forests        | Yes         | Yes          |
    When structural gap analysis runs
    Then it should identify:
      | Gap_Type              | Concept         | Issue                    |
      | Missing_Prerequisites | Decision_Trees  | No foundation defined    |
      | Orphan_Island         | -               | None detected            |
      | Weak_Connectivity     | -               | None detected            |
    And recommendation: "Add prerequisites for Decision_Trees"

  # ─────────────────────────────────────────────────────────────────────────
  # Atom-Concept Relationships
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Graph @AtomLinking
  Scenario: Linking atoms to concepts with coverage metadata
    Given concept "Recursion" requires coverage of:
      | Aspect              | Required_Coverage |
      | Definition          | 1.0               |
      | Base_Case           | 1.0               |
      | Recursive_Case      | 1.0               |
      | Stack_Behavior      | 0.8               |
      | Tail_Optimization   | 0.5               |
    When atoms are linked with coverage:
      | Atom_ID | Type        | Covers                        | Coverage |
      | A1      | Flashcard   | Definition                    | 1.0      |
      | A2      | MCQ         | Base_Case, Recursive_Case     | 0.5 each |
      | A3      | Parsons     | Stack_Behavior                | 0.8      |
      | A4      | Socratic    | All_Aspects                   | 0.3 each |
    Then concept coverage should compute to:
      | Aspect              | Total_Coverage | Status   |
      | Definition          | 1.0 + 0.3      | Complete |
      | Base_Case           | 0.5 + 0.3      | Complete |
      | Recursive_Case      | 0.5 + 0.3      | Complete |
      | Stack_Behavior      | 0.8 + 0.3      | Complete |
      | Tail_Optimization   | 0.3            | Gap      |
    And flag: "Tail_Optimization needs additional atoms"

  @Positive @Graph @AtomICAPDistribution
  Scenario: Ensuring ICAP mode distribution for concepts
    Given concept "Binary Search" has atoms:
      | ICAP_Mode    | Count |
      | Passive      | 5     |
      | Active       | 8     |
      | Constructive | 3     |
      | Interactive  | 0     |
    When ICAP distribution analyzer runs
    Then it should identify:
      | Issue                 | Details                        |
      | Missing_Interactive   | 0 Interactive atoms            |
      | Heavy_Active          | 50% of atoms are Active        |
    And recommend: "Create Interactive atoms for deeper engagement"

  # ─────────────────────────────────────────────────────────────────────────
  # Performance and Caching
  # ─────────────────────────────────────────────────────────────────────────

  @Positive @Performance @Caching
  Scenario: Caching frequently accessed subgraphs
    Given concept "Python Basics" is accessed 1000 times/day
    When the caching layer analyzes access patterns
    Then it should cache:
      | Cached_Data           | TTL      | Invalidation_Trigger     |
      | Prerequisite_Chain    | 1 hour   | Any prereq edge change   |
      | Teaching_Atoms        | 30 min   | Atom CRUD operation      |
      | Centrality_Scores     | 24 hours | Weekly recalculation     |
    And cache hit rate should exceed 90% for hot concepts

  @Positive @Performance @BatchOperations
  Scenario: Efficient batch operations for curriculum import
    Given a curriculum import with 200 concepts and 500 edges
    When batch import executes
    Then it should:
      | Optimization          | Details                        |
      | Use_UNWIND            | Batch node creation            |
      | Defer_Indexing        | Index after bulk insert        |
      | Transaction_Batching  | 50 items per transaction       |
    And total import time should be < 30 seconds
    And rollback should be atomic on failure

