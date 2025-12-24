import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class EvidenceItem:
    file_name: str
    domain: str
    content: Dict[str, Any]
    path: str

class EvidenceConnector:
    """
    Connects to the Research Engine's evidence repository.
    Reads verified evidence files for indexing into the Knowledge Graph.
    """
    
    def __init__(self, evidence_root: str):
        # Treat empty string as "not provided" -> force fallback
        if not evidence_root:
             self.evidence_root = Path("non_existent_path_to_force_fallback")
        else:
             self.evidence_root = Path(evidence_root)
             
        if not self.evidence_root.exists():
            # Fallback for relative paths in monorepo
            repo_root = Path("e:/Repo/project-astartes")
            self.evidence_root = repo_root / "services/core/research-engine/evidence"
            
            # If still doesn't exist (e.g. running on different machine), warn
            if not self.evidence_root.exists():
                print(f"Warning: Evidence root could not be located at {self.evidence_root}")
    
    def list_evidence_files(self, domain: Optional[str] = None) -> List[Path]:
        """List all .json evidence files, optionally filtered by domain."""
        if not self.evidence_root.exists():
            print(f"Warning: Evidence root {self.evidence_root} does not exist.")
            return []
            
        patterns = [f"{domain}/*.json"] if domain else ["*/*.json"]
        files = []
        for pattern in patterns:
            files.extend(list(self.evidence_root.glob(pattern)))
        return files

    def read_evidence(self, file_path: Path) -> Optional[EvidenceItem]:
        """Read and parse a single evidence file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            domain = file_path.parent.name
            return EvidenceItem(
                file_name=file_path.name,
                domain=domain,
                content=content,
                path=str(file_path)
            )
        except Exception as e:
            print(f"Error reading evidence {file_path}: {e}")
            return None

    def get_all_evidence(self) -> List[EvidenceItem]:
        """Retrieve all accessible evidence items."""
        files = self.list_evidence_files()
        items = []
        for f in files:
            item = self.read_evidence(f)
            if item:
                items.append(item)
        return items
