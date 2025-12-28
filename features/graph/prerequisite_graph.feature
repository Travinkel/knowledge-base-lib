@KnowledgeBase @Graph @Prerequisites @Priority-High
Feature: Prerequisite Graph Operations
  As an adaptive engine
  I want to query prerequisite relationships
  So that learning paths respect concept dependencies

  Background:
    Given the Knowledge Base service is initialized
    And the graph storage backend is connected

  # =========================================================
  # PREREQUISITE CRUD
  # =========================================================

  @Smoke @CRUD
  Scenario: Add a required prerequisite
    Given atom "VLSM" exists in the knowledge base
    And atom "Subnetting" exists in the knowledge base
    When I add prerequisite from "VLSM" to "Subnetting" with strength "required"
    Then the prerequisite link should be created
    And the link should have strength "required"
    And the link should have default confidence 0.5

  @Smoke @CRUD
  Scenario: Add a recommended prerequisite with rationale
    Given atom "Network Design" exists in the knowledge base
    And atom "VLSM" exists in the knowledge base
    When I add prerequisite from "Network Design" to "VLSM" with:
      | strength   | recommended                              |
      | rationale  | VLSM enables efficient IP address usage |
      | confidence | 0.85                                    |
    Then the prerequisite link should be created
    And the link rationale should be "VLSM enables efficient IP address usage"
    And the link confidence should be 0.85

  @Regression @CRUD
  Scenario: Remove a prerequisite
    Given atom "A" has prerequisite "B"
    When I remove the prerequisite from "A" to "B"
    Then the prerequisite link should be deleted
    And "A" should have no prerequisites

  # =========================================================
  # PREREQUISITE QUERIES
  # =========================================================

  @Smoke @Query
  Scenario: Query direct prerequisites
    Given atom "VLSM" with prerequisites:
      | prerequisite  | strength    |
      | Subnetting    | required    |
      | Binary Math   | required    |
      | IP Addressing | recommended |
    When I query prerequisites for "VLSM"
    Then I should get 3 prerequisite links
    And "Subnetting" should be a "required" prerequisite
    And "Binary Math" should be a "required" prerequisite
    And "IP Addressing" should be a "recommended" prerequisite

  @Regression @Query
  Scenario: Query prerequisites filtered by strength
    Given atom "VLSM" with prerequisites:
      | prerequisite  | strength    |
      | Subnetting    | required    |
      | Binary Math   | required    |
      | IP Addressing | recommended |
    When I query prerequisites for "VLSM" with strength "required"
    Then I should get 2 prerequisite links
    And "IP Addressing" should not be in the results

  @Regression @Query
  Scenario: Query transitive prerequisites
    Given the following prerequisite chain:
      | from    | to          |
      | VLSM    | Subnetting  |
      | Subnetting | IP Addressing |
      | IP Addressing | Binary Basics |
    When I query prerequisites for "VLSM" with transitive=true
    Then I should get 3 prerequisite links
    And the results should include "Subnetting", "IP Addressing", "Binary Basics"

  @Smoke @Query
  Scenario: Query dependents (reverse lookup)
    Given atoms "A", "B", "C" where "B" and "C" require "A"
    When I query dependents for "A"
    Then I should get 2 dependent atoms
    And the results should include "B" and "C"

  # =========================================================
  # DAG VALIDATION
  # =========================================================

  @Smoke @DAG
  Scenario: Prerequisite graph is acyclic
    Given a set of prerequisite links forming a DAG
    When I validate the prerequisite graph
    Then no cycles should be detected
    And validation should return is_valid=true

  @Regression @DAG
  Scenario: Detect cycle in prerequisites
    Given atoms "A", "B", "C" with prerequisites:
      | from | to |
      | A    | B  |
      | B    | C  |
    When I attempt to add prerequisite from "C" to "A"
    Then a CycleDetectedError should be raised
    And the error should contain the cycle path

  @Regression @DAG
  Scenario: Topological sort succeeds for valid DAG
    Given the following prerequisite chain:
      | from      | to        |
      | Advanced  | Intermediate |
      | Intermediate | Basics   |
    When I request topological sort
    Then the result should be ["Basics", "Intermediate", "Advanced"]
    And prerequisites should come before dependents

  # =========================================================
  # LEARNING PATHS
  # =========================================================

  @Smoke @LearningPath
  Scenario: Get learning path to target atom
    Given the following prerequisite chain:
      | from    | to          |
      | VLSM    | Subnetting  |
      | Subnetting | IP Basics |
    And I have mastered no atoms
    When I request learning path to "VLSM"
    Then the path should be ["IP Basics", "Subnetting", "VLSM"]
    And each atom should precede its dependents

  @Regression @LearningPath
  Scenario: Learning path excludes mastered atoms
    Given the following prerequisite chain:
      | from    | to          |
      | VLSM    | Subnetting  |
      | Subnetting | IP Basics |
    And I have mastered ["IP Basics", "Subnetting"]
    When I request learning path to "VLSM"
    Then the path should be ["VLSM"]
    And mastered atoms should be excluded

  @Smoke @LearningPath
  Scenario: Get next atoms to learn
    Given atoms with prerequisites:
      | atom   | prerequisites          |
      | A      | []                     |
      | B      | ["A"]                  |
      | C      | ["A"]                  |
      | D      | ["B", "C"]             |
    And I have mastered ["A"]
    When I request next atoms to learn
    Then the results should include "B" and "C"
    And "D" should not be in the results (missing prereqs)

  @Regression @LearningPath
  Scenario: Next atoms prioritizes foundational concepts
    Given atoms with many dependents:
      | atom        | dependent_count |
      | Foundational| 10              |
      | Specific    | 1               |
    And both have their prerequisites met
    When I request next atoms with limit 1
    Then "Foundational" should be first (more dependents)
