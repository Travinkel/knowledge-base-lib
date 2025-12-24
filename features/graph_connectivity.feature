@Domain-KnowledgeBase @Infrastructure-GraphDB @Priority-High
Feature: Knowledge Graph Connectivity
  As a System Architect
  I want the Knowledge Base to connect to a Graph Database (Neo4j)
  So that I can query complex semantic relationships efficiently without local file systems.

  Background:
    Given the GraphConnector is configured with Neo4j credentials
    And the Neo4j driver is active and verified

  @Positive @Smoke
  Scenario: Establish Connection and Retrieve Atom
    Given a Learning Atom "Atom_123" exists in the GraphDB
    When the system requests "Atom_123" via the GraphConnector
    Then it should return a "Node" object with properties (title, type, content)
    And the retrieval latency should be less than 50ms

  @Positive @Query
  Scenario: Traverse Prerequisite Path
    Given "Atom_B" depends on "Atom_A" with a "HARD_PREREQUISITE" edge
    When the system queries the learning path for "Atom_B"
    Then it should return a path sequence ["Atom_A", "Atom_B"]
    And the edge metadata should include "weight" and "confidence"

  @Negative @ErrorHandling
  Scenario: Handle Database Unavailability
    Given the Neo4j instance is unreachable (simulated)
    When the system attempts to push a new atom
    Then it should raise a "GraphConnectionError"
    And it should cache the write operation in a local "RetryQueue"
