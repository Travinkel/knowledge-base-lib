import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Try to import neo4j, but don't fail if not installed in this environment
# (This allows running in environments where only mocks are available)
try:
    from neo4j import GraphDatabase, Driver
except ImportError:
    GraphDatabase = None
    Driver = None

@dataclass
class GraphNode:
    id: str
    labels: List[str]
    properties: Dict[str, Any]

class GraphConnectionError(Exception):
    pass

class GraphConnector:
    """
    Connects to Neo4j Graph Database for Knowledge Base operations.
    Replaces the file-based EvidenceConnector.
    """

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        if GraphDatabase:
            try:
                self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            except Exception as e:
                print(f"Warning: Failed to initialize Neo4j driver: {e}")
                self.driver = None
        else:
            print("Warning: Neo4j library not found. Running in driverless/mock mode.")

    def close(self):
        if self.driver:
            self.driver.close()

    def get_atom(self, atom_id: str) -> Optional[GraphNode]:
        """Retrieve a single atom node by ID."""
        if not self.driver:
            raise GraphConnectionError("Neo4j driver not active")

        query = "MATCH (n:LearningAtom {id: $atom_id}) RETURN n"
        with self.driver.session() as session:
            result = session.run(query, atom_id=atom_id)
            record = result.single()
            if record:
                node = record["n"]
                return GraphNode(
                    id=node.get("id"),
                    labels=list(node.labels),
                    properties=dict(node)
                )
        return None

    def get_learning_path(self, target_atom_id: str) -> List[Dict[str, Any]]:
        """Retrieve prerequisite path for a target atom."""
        if not self.driver:
            raise GraphConnectionError("Neo4j driver not active")
        
        # Simple BFS/shortest path query for prerequisites
        query = """
        MATCH p = shortestPath((root:LearningAtom)-[:PREREQUISITE*]->(target:LearningAtom {id: $id}))
        RETURN nodes(p) as path_nodes, relationships(p) as path_edges
        """
        path_sequence = []
        with self.driver.session() as session:
            result = session.run(query, id=target_atom_id)
            record = result.single()
            if record:
                # Naive serialization of the path
                nodes = record["path_nodes"]
                for n in nodes:
                   path_sequence.append({
                       "id": n.get("id"),
                       "title": n.get("title", "Unknown")
                   })
        return path_sequence

    def push_atom(self, atom_data: Dict[str, Any]):
        """Upsert an atom node."""
        if not self.driver:
            raise GraphConnectionError("Neo4j driver not active")

        query = """
        MERGE (n:LearningAtom {id: $id})
        SET n += $props
        RETURN n
        """
        # Separate ID from other props
        atom_id = atom_data.get("id")
        if not atom_id:
            raise ValueError("Atom data must have an 'id'")
            
        with self.driver.session() as session:
            try:
                session.run(query, id=atom_id, props=atom_data)
            except Exception as e:
                raise GraphConnectionError(f"Failed to push atom: {e}")

