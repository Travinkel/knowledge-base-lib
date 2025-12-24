@Regression @Domain:Database @Logic:Positive @Status:Stable
Feature: Platinum-Layer Knowledge Base Integration
  Goal: Ensure seamless integration of learning atoms into the persistent PostgreSQL Knowledge Base for long-term mastery tracking.

  Background:
    Given the ETL pipeline has generated a set of "Gold-Layer Atoms"
    And the PostgreSQL "Master Ledger" schema is synchronized
    And the Semantic Linker agent is active

  Scenario: Deep Semantic Linking (Platinum Layer)
    When a "Binary Search" Gold-Layer Atom is ingested
    Then the system shall create a "Semantic Link" to "Divide and Conquer" (Parent Concept)
    And it shall create a "Prerequisite Link" to "Array Indexing" and "Logarithmic Functions"
    And it shall store these links in the `knowledge_graph_edges` table with a "Conceptual Strength" weight.

  Scenario: NCDE Feedback Loop
    Given a user has failed a "Red-Black Tree Insertion" Atom 3 times
    When the system updates the "Master Ledger" via MCP
    Then the "Struggle_Signal" shall trigger a "Knowledge Base Query" for "Bridge Atoms"
    And the system shall retrieve a "Physical Analogy" atom (e.g., "Library Shelf Rebalancing") to reset the Base Domain.

  Scenario Outline: Knowledge Base Retrieval Patterns
    When the cortex-cli requests an atom for "<Concept>" at "<MasteryLevel>"
    Then the Knowledge Base shall return an atom from the "<TargetLayer>"
    And the selection shall prioritize atoms with "<Attribute>"

    Examples:
      | Concept            | MasteryLevel | TargetLayer | Attribute             |
      | Matrix Calculus    | Novice       | Gold        | Analogical Bridge     |
      | Garbage Collection | Expert       | Gold        | Socratic Scaffolding  |
      | Fourier Transform  | Ramanujan    | Platinum    | Formal Verification   |
      | Huffman Coding     | Competent    | Gold        | Productive Failure    |
