from __future__ import annotations

import ast
from pathlib import Path
import unittest


PROJECT_DIR = Path(__file__).resolve().parents[1]


class LegacyRestoreBaselineTests(unittest.TestCase):
    def test_numerical_functions_still_match_reference_ast(self) -> None:
        reference = PROJECT_DIR / "ref/restore.py"
        copied = PROJECT_DIR / "src/restoration/legacy/restore.py"
        if not reference.is_file():
            self.skipTest("local reference data are not available")
        reference_tree = ast.parse(reference.read_text(encoding="utf-8"))
        adapted_tree = ast.parse(copied.read_text(encoding="utf-8"))

        def functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
            return {
                node.name: node
                for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }

        reference_functions = functions(reference_tree)
        adapted_functions = functions(adapted_tree)
        for name in (
            "save_reference_restoration",
            "run_cfg",
            "run_cfg_lin_cut",
        ):
            with self.subTest(function=name):
                self.assertEqual(
                    ast.dump(adapted_functions[name], include_attributes=False),
                    ast.dump(reference_functions[name], include_attributes=False),
                )

    def test_hard_coded_configuration_builders_are_removed(self) -> None:
        copied = PROJECT_DIR / "src/restoration/legacy/restore.py"
        tree = ast.parse(copied.read_text(encoding="utf-8"))
        function_names = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        self.assertNotIn("make_cfg_arr", function_names)
        self.assertNotIn("make_cfg_arr_from_params", function_names)
        self.assertIn("run_legacy_restore", function_names)
