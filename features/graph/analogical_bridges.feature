@KnowledgeBase @Graph @Analogies @Priority-Medium
Feature: Analogical Bridge Queries
  As a tutoring agent
  I want to find analogies for concepts
  So that I can scaffold learning with familiar examples

  Background:
    Given the Knowledge Base service is initialized
    And the graph storage backend is connected
    And the vector storage backend is connected

  # =========================================================
  # BRIDGE CRUD
  # =========================================================

  @Smoke @CRUD
  Scenario: Create a structural analogy bridge
    Given atom "IP Address" in domain "Networking"
    And atom "Street Address" in domain "Everyday Life"
    When I create an analogical bridge with:
      | source      | IP Address     |
      | target      | Street Address |
      | bridge_type | structural     |
      | similarity  | 0.85           |
    Then the bridge should be created
    And the bridge type should be "structural"

  @Smoke @CRUD
  Scenario: Create bridge with structural mappings
    Given atom "Stack (Data Structure)" in domain "Computer Science"
    And atom "Cafeteria Tray Pile" in domain "Everyday Life"
    When I create an analogical bridge with mappings:
      | source     | target          |
      | push()     | add tray        |
      | pop()      | remove tray     |
      | LIFO       | last in first   |
      | isEmpty()  | pile empty      |
    Then the bridge should have 4 structural mappings
    And the bridge type should be "structural"

  @Regression @CRUD
  Scenario: Remove an analogical bridge
    Given an existing bridge from "A" to "B"
    When I remove the analogy from "A" to "B"
    Then the bridge should be deleted
    And "A" should have no analogies

  # =========================================================
  # BRIDGE QUERIES
  # =========================================================

  @Smoke @Query
  Scenario: Find bridges for concept
    Given atom "IP Address" with bridges to:
      | target         | bridge_type | similarity |
      | Street Address | structural  | 0.85       |
      | Phone Number   | surface     | 0.45       |
    When I query bridges for "IP Address"
    Then I should get 2 bridges
    And "Street Address" should have similarity 0.85
    And "Phone Number" should have similarity 0.45

  @Regression @Query
  Scenario: Filter bridges by type
    Given atom "Binary Search" with bridges:
      | target           | bridge_type |
      | Number Guessing  | structural  |
      | Finding in Book  | surface     |
    When I query bridges for "Binary Search" with type "structural"
    Then I should get 1 bridge
    And the target should be "Number Guessing"

  @Regression @Query
  Scenario: Filter bridges by minimum similarity
    Given atom "Recursion" with bridges:
      | target        | similarity |
      | Russian Dolls | 0.90       |
      | Mirror        | 0.40       |
    When I query bridges for "Recursion" with min_similarity 0.5
    Then I should get 1 bridge
    And the target should be "Russian Dolls"

  # =========================================================
  # ANALOGY DISCOVERY
  # =========================================================

  @Smoke @Discovery
  Scenario: Find similar atoms using vector search
    Given atom "Binary Search" with embedding
    And 100 other atoms with embeddings across domains
    When I search for similar atoms to "Binary Search" with cross_domain=true
    Then results should prioritize atoms from different domains
    And each result should have similarity score >= 0.5

  @Regression @Discovery
  Scenario: Suggest analogies for review
    Given atom "Hash Table" in domain "Computer Science"
    And atoms in domain "Library Science":
      | atom              |
      | Card Catalog      |
      | Dewey Decimal     |
    When I request analogy suggestions for "Hash Table"
    Then suggestions should include potential bridges
    And each suggestion should have computed structural mappings
    And suggestions should be sorted by match quality

  # =========================================================
  # CROSS-DOMAIN BRIDGING
  # =========================================================

  @Smoke @CrossDomain
  Scenario: Find atoms bridging two domains
    Given bridges between "Computer Science" and "Biology":
      | source           | target          | similarity |
      | Neural Network   | Brain           | 0.88       |
      | Virus (malware)  | Virus (bio)     | 0.75       |
      | Memory           | Memory (brain)  | 0.70       |
    When I query bridges from "Computer Science" to "Biology"
    Then I should get bridges sorted by similarity
    And the first result should be "Neural Network" -> "Brain"

  @Regression @CrossDomain
  Scenario: No bridges between unrelated domains
    Given atoms in "Networking" and "Cooking"
    And no analogical bridges between them
    When I query bridges from "Networking" to "Cooking"
    Then I should get empty results

  # =========================================================
  # STRUCTURE MAPPING
  # =========================================================

  @Smoke @StructureMapping @evidence:gentner1983
  Scenario: Validate structural bridge has mappings
    Given a structural analogy between:
      | source concept | Solar System |
      | target concept | Atom         |
    When the bridge is created
    Then structural_mappings should include:
      | source      | target     |
      | sun         | nucleus    |
      | planets     | electrons  |
      | orbit       | orbit      |
      | gravity     | electromagnetism |
    And the bridge should capture relational similarity

  @Regression @StructureMapping
  Scenario: Surface bridge has fewer mappings
    Given a surface analogy between:
      | source concept | Red Apple  |
      | target concept | Red Ball   |
    When the bridge is created
    Then structural_mappings should only include:
      | source | target |
      | red    | red    |
      | round  | round  |
    And the bridge_type should be "surface"
    And similarity should be lower than structural bridges

  # =========================================================
  # ANALOGY APPLICATION
  # =========================================================

  @Smoke @Application
  Scenario: Use analogy to scaffold unfamiliar concept
    Given learner is familiar with "Street Address" domain
    And learner is learning "IP Address" in "Networking"
    And there's a bridge from "IP Address" to "Street Address"
    When I request scaffolding for "IP Address"
    Then the response should reference "Street Address" analogy
    And structural mappings should guide explanation
