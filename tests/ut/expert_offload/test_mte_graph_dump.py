# SPDX-License-Identifier: Apache-2.0
import ast
import unittest
from pathlib import Path


class GraphDumpPlacementTests(unittest.TestCase):
    def test_dump_is_outside_capture_and_guarded(self):
        path = Path(__file__).resolve().parents[3] / "vllm_ascend/compilation/acl_graph.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "debug_dump"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 1, "Torch NPU 2.10 requires debug_path")
        ancestors = []
        node = calls[0]
        while node in parents:
            node = parents[node]
            ancestors.append(node)
        self.assertFalse(any(isinstance(node, ast.With) for node in ancestors))
        self.assertTrue(
            any(
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Attribute)
                and node.test.attr == "_moe_graph_dump"
                for node in ancestors
            )
        )


if __name__ == "__main__":
    unittest.main()
