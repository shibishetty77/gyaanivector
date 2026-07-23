"""
Rules for extracting entities and relationships from the YAML metadata.
"""

from typing import Any


def _as_list(value: Any) -> list[str]:
    """Normalize a scalar or list to a list of strings."""
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def extract_graph_elements(metadata: dict[str, Any]) -> tuple[str | None, list[str], list[tuple[str, str, str]]]:
    """
    Given a parsed YAML dictionary, extract:
      - The primary node ID for this document.
      - A list of aliases/synonyms for the node (used for query matching).
      - A list of (source, relation, target) edges.
    """
    edges: list[tuple[str, str, str]] = []
    
    # 1. Determine primary entity ID
    node_id = None
    for key in ("id", "evaluation_id"):
        if key in metadata and metadata[key]:
            node_id = str(metadata[key]).lower().strip()
            break
            
    if not node_id:
        # If no explicit ID, we can't reliably build edges originating from this doc
        return None, [], []

    # 2. Collect aliases (e.g., node_id itself, title, library names)
    aliases: set[str] = {node_id}
    if "title" in metadata:
        aliases.add(str(metadata["title"]).lower().strip())
    
    # Add any explicit aliases if they exist in the schema
    for alias in _as_list(metadata.get("aliases")):
        aliases.add(alias.lower().strip())

    # 3. Extract relationships based on schema fields
    
    # Class composition
    for cls in _as_list(metadata.get("class")):
        edges.append((node_id, "is_a", cls.lower().strip()))
        
    # Interfaces (e.g., uses I2C, uses SPI)
    for iface in _as_list(metadata.get("interface")):
        edges.append((node_id, "uses_interface", iface.lower().strip()))
        
    # Hardware compatibility
    for compat in _as_list(metadata.get("compatible_with")):
        edges.append((node_id, "compatible_with", compat.lower().strip()))
        
    # Facts/Classes used (from evaluation sets)
    for fact in _as_list(metadata.get("facts_used")):
        edges.append((node_id, "tests_fact", fact.lower().strip()))
        
    for cls in _as_list(metadata.get("classes_used")):
        edges.append((node_id, "tests_class", cls.lower().strip()))

    return node_id, list(aliases), edges
