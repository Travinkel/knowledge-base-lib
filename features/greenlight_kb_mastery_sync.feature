@Regression @Domain:Greenlight @Domain:Database @Logic:Adaptive @Status:Stable
Feature: Greenlight-KB Mastery Sync
  Goal: Ensure that performance in the IDE (Greenlight) is accurately reflected in the Knowledge Base (Master Ledger) and used for subsequent session planning.

  Background:
    Given the learner is working on a "Code_Submission" atom in Greenlight
    And the Greenlight-Client is connected to the cortex-cli via MCP

  Scenario: Successful Code Submission Sync
    When the learner passes all unit tests in Greenlight for "Async_Await_Pattern"
    Then Greenlight shall return a "Success" payload to the cortex-cli
    And the cortex-cli shall update the KnowledgeBase `learner_skill_mastery` for the "Asynchrony" skill
    And the FSRS algorithm shall adjust the next "Hardening" interval to 14 days
    And the AstartesAgents shall log a "Procedural Mastery" event for the user.

  Scenario: Failure with Misconception Diagnosis
    When the learner fails the "Resource_Cleanup" test in Greenlight
    And the Greenlight-Audit agent identifies a "Try-Finally-Missing" error pattern
    Then the cortex-cli shall fetch a "Socratic_Remediation_Atom" from the KnowledgeBase
    And the Spivak Agent shall initiate a "Reflection Prompt" to bridge the gap between "Manual Cleanup" and "Deterministic Finalization."

  Scenario Outline: Adaptive Pathing Based on Greenlight Metrics
    Given the user's "<Metric>" in Greenlight is "<Value>"
    When the session returns control to cortex-cli
    Then the system shall "<Action>" the learning path in the KnowledgeBase
    And the Spivak Agent shall provide a "<Narrative>" feedback.

    Examples:
      | Metric            | Value    | Action               | Narrative                       |
      | TestPassRate      | 1.0      | Skip_Intermediate    | Accelerated pathing enabled.    |
      | ExecutionTime     | > 2x     | Inject_Optimization  | Focus on algorithmic efficiency.|
      | CyclomaticComp    | High     | Inject_Refactoring   | Simplifying mental model...     |
      | Struggle_Signal   | Detected | Revert_to_Bridge     | Returning to analogical base.   |
