"""Wikimedia Commons Category Directed Acyclic Graph (DAG) with Tarjan's SCC cycle handling."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Set, List, Optional, Tuple, Any

from src.data.category_vocab import CategoryVocabulary


class CategoryDAG:
    """Directed graph representing parent-child category relationships with cycle resolution."""

    def __init__(self, vocab: Optional[CategoryVocabulary] = None):
        self.vocab = vocab
        # Adjacency representations:
        # parent_to_children[u] = set of v where u is parent of v (v is subcategory of u)
        self.parent_to_children: Dict[int, Set[int]] = {}
        # child_to_parents[v] = set of u where u is parent of v
        self.child_to_parents: Dict[int, Set[int]] = {}
        self.cat_names: Dict[int, str] = {}

    def add_category(self, category_id: int, name: str) -> None:
        """Registers a category node in the graph."""
        self.cat_names[category_id] = name
        if category_id not in self.parent_to_children:
            self.parent_to_children[category_id] = set()
        if category_id not in self.child_to_parents:
            self.child_to_parents[category_id] = set()

    def add_edge(self, parent_id: int, child_id: int) -> None:
        """Adds a directed taxonomic edge from parent to child (child is a subcategory of parent)."""
        if parent_id not in self.parent_to_children:
            self.parent_to_children[parent_id] = set()
        if parent_id not in self.child_to_parents:
            self.child_to_parents[parent_id] = set()

        if child_id not in self.parent_to_children:
            self.parent_to_children[child_id] = set()
        if child_id not in self.child_to_parents:
            self.child_to_parents[child_id] = set()

        self.parent_to_children[parent_id].add(child_id)
        self.child_to_parents[child_id].add(parent_id)

    def get_children(self, category_id: int) -> Set[int]:
        """Returns direct subcategories of category_id."""
        return self.parent_to_children.get(category_id, set())

    def get_parents(self, category_id: int) -> Set[int]:
        """Returns direct parent categories of category_id."""
        return self.child_to_parents.get(category_id, set())

    def get_descendants(self, category_id: int, max_depth: int = 15) -> Set[int]:
        """Traverses the graph downward to return all descendant category IDs (cycle-safe)."""
        descendants: Set[int] = set()
        visited: Set[int] = {category_id}
        queue: List[Tuple[int, int]] = [(category_id, 0)]

        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for child in self.parent_to_children.get(curr, ()):
                if child not in visited:
                    visited.add(child)
                    descendants.add(child)
                    queue.append((child, depth + 1))

        return descendants

    def get_ancestors(self, category_id: int, max_depth: int = 15) -> Set[int]:
        """Traverses the graph upward to return all ancestor category IDs (cycle-safe)."""
        ancestors: Set[int] = set()
        visited: Set[int] = {category_id}
        queue: List[Tuple[int, int]] = [(category_id, 0)]

        while queue:
            curr, depth = queue.pop(0)
            if depth >= max_depth:
                continue

            for parent in self.child_to_parents.get(curr, ()):
                if parent not in visited:
                    visited.add(parent)
                    ancestors.add(parent)
                    queue.append((parent, depth + 1))

        return ancestors

    def find_strongly_connected_components(self) -> List[List[int]]:
        """Identifies cycles using Tarjan's Strongly Connected Components algorithm in O(V + E)."""
        index = 0
        indices: Dict[int, int] = {}
        lowlinks: Dict[int, int] = {}
        on_stack: Dict[int, bool] = {}
        stack: List[int] = []
        sccs: List[List[int]] = []

        all_nodes = set(self.parent_to_children.keys()) | set(self.child_to_parents.keys())

        def strongconnect(v: int):
            nonlocal index
            indices[v] = index
            lowlinks[v] = index
            index += 1
            stack.append(v)
            on_stack[v] = True

            for w in self.parent_to_children.get(v, ()):
                if w not in indices:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack.get(w, False):
                    lowlinks[v] = min(lowlinks[v], indices[w])

            if lowlinks[v] == indices[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)

        for node in all_nodes:
            if node not in indices:
                strongconnect(node)

        return sccs

    def resolve_cycles(self) -> int:
        """Removes cyclic back-edges based on SCCs to guarantee an acyclic DAG."""
        sccs = self.find_strongly_connected_components()
        removed_edges = 0

        for scc in sccs:
            if len(scc) > 1:
                # Cycle detected: break back-edge from the last node to the first
                scc_set = set(scc)
                for u in scc:
                    for v in list(self.parent_to_children.get(u, ())):
                        if v in scc_set and u != v:
                            # Prune edge within component to make acyclic
                            self.parent_to_children[u].remove(v)
                            if u in self.child_to_parents.get(v, ()):
                                self.child_to_parents[v].remove(u)
                            removed_edges += 1
                            break

        return removed_edges

    def save(self, filepath: str | Path) -> None:
        """Serializes the category DAG to JSON."""
        fp = Path(filepath)
        fp.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "categories": [{"id": k, "name": v} for k, v in self.cat_names.items()],
            "edges": [
                {"parent": parent, "child": child}
                for parent, children in self.parent_to_children.items()
                for child in children
            ],
        }
        with open(fp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path, vocab: Optional[CategoryVocabulary] = None) -> CategoryDAG:
        """Loads category DAG from JSON."""
        dag = cls(vocab=vocab)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        for cat in data.get("categories", []):
            dag.add_category(cat["id"], cat["name"])

        for edge in data.get("edges", []):
            dag.add_edge(edge["parent"], edge["child"])

        return dag
