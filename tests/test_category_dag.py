"""Unit tests for CategoryDAG and Tarjan's SCC cycle handling."""

import unittest
import tempfile
from pathlib import Path
from src.stage2.graph.category_dag import CategoryDAG


class TestCategoryDAG(unittest.TestCase):
    def setUp(self):
        self.dag = CategoryDAG()
        # 0: Architecture
        # 1: Religious architecture
        # 2: Cathedrals
        # 3: Gothic cathedrals
        self.dag.add_category(0, "Category:Architecture")
        self.dag.add_category(1, "Category:Religious architecture")
        self.dag.add_category(2, "Category:Cathedrals")
        self.dag.add_category(3, "Category:Gothic cathedrals")

        self.dag.add_edge(0, 1)  # Architecture -> Religious architecture
        self.dag.add_edge(1, 2)  # Religious architecture -> Cathedrals
        self.dag.add_edge(2, 3)  # Cathedrals -> Gothic cathedrals

    def test_parents_and_children(self):
        self.assertEqual(self.dag.get_children(0), {1})
        self.assertEqual(self.dag.get_children(2), {3})
        self.assertEqual(self.dag.get_parents(3), {2})
        self.assertEqual(self.dag.get_parents(0), set())

    def test_ancestors_and_descendants(self):
        desc = self.dag.get_descendants(0)
        self.assertEqual(desc, {1, 2, 3})

        anc = self.dag.get_ancestors(3)
        self.assertEqual(anc, {0, 1, 2})

    def test_tarjan_scc_and_cycle_resolution(self):
        # Create a graph with a cycle: 10 -> 20 -> 30 -> 10
        cyclic_dag = CategoryDAG()
        for i in [10, 20, 30, 40]:
            cyclic_dag.add_category(i, f"Category:{i}")

        cyclic_dag.add_edge(10, 20)
        cyclic_dag.add_edge(20, 30)
        cyclic_dag.add_edge(30, 10)  # Cycle!
        cyclic_dag.add_edge(30, 40)  # Branch out

        sccs = cyclic_dag.find_strongly_connected_components()
        multi_node_sccs = [scc for scc in sccs if len(scc) > 1]
        self.assertEqual(len(multi_node_sccs), 1)
        self.assertEqual(set(multi_node_sccs[0]), {10, 20, 30})

        # Resolve cycle
        removed = cyclic_dag.resolve_cycles()
        self.assertGreater(removed, 0)

        # After resolution, no multi-node SCCs should remain
        sccs_after = cyclic_dag.find_strongly_connected_components()
        multi_node_after = [scc for scc in sccs_after if len(scc) > 1]
        self.assertEqual(len(multi_node_after), 0)

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "dag.json"
            self.dag.save(file_path)

            loaded = CategoryDAG.load(file_path)
            self.assertEqual(len(loaded.cat_names), 4)
            self.assertEqual(loaded.get_children(0), {1})
            self.assertEqual(loaded.get_descendants(0), {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
