"""
Test runner for Knowledge Graph Operations BDD scenarios.

This file is the entry point for pytest-bdd to discover and run
the Gherkin scenarios from graph_operations.feature.

Run with:
    pytest test_graph_operations.py -v
"""

import pytest
from pytest_bdd import scenarios

# Import step definitions - this registers all steps
from step_defs.graph_operations_steps import *

# Link feature file to this test module
# The scenarios decorator imports all scenarios from the feature file
scenarios("../features/graph_operations.feature")
