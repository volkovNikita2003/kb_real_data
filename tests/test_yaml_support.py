from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import tempfile
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from yaml_support import (
    YAML_OMIT_NONE,
    YamlError,
    dump_yaml,
    load_yaml,
    to_plain_data,
)


@dataclass(frozen=True)
class Example:
    name: str
    values: tuple[int, ...]
    path: Path


@dataclass(frozen=True)
class OptionalExample:
    omitted: str | None = field(
        default=None, metadata={YAML_OMIT_NONE: True}
    )
    retained: str | None = None


class LoadYamlTests(unittest.TestCase):
    def test_loads_safe_yaml_and_preserves_unicode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.yaml"
            path.write_text("название: калибровка\nvalue: 2\n", encoding="utf-8")

            self.assertEqual(
                load_yaml(path),
                {"название": "калибровка", "value": 2},
            )

    def test_rejects_duplicate_keys_at_any_level(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.yaml"
            path.write_text(
                "root:\n  value: 1\n  value: 2\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(YamlError, "duplicate key 'value'"):
                load_yaml(path)

    def test_rejects_unsafe_python_tags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text(
                "value: !!python/object/apply:builtins.str [unsafe]\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(YamlError, "Не удалось прочитать YAML"):
                load_yaml(path)

    def test_rejects_missing_empty_directory_and_malformed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (root / "missing.yaml", "не найден"),
                (root / "empty.yaml", "пуст"),
                (root / "subdirectory", "не является файлом"),
                (root / "malformed.yaml", "Не удалось прочитать YAML"),
            )
            (root / "empty.yaml").write_text("", encoding="utf-8")
            (root / "subdirectory").mkdir()
            (root / "malformed.yaml").write_text("value: [1\n", encoding="utf-8")

            for path, message in cases:
                with self.subTest(path=path.name), self.assertRaisesRegex(
                    YamlError, message
                ):
                    load_yaml(path)


class DumpYamlTests(unittest.TestCase):
    def test_only_marked_none_dataclass_fields_are_omitted(self) -> None:
        self.assertEqual(
            to_plain_data(OptionalExample()),
            {"retained": None},
        )

    def test_plain_conversion_handles_dataclasses_mappings_tuples_and_paths(self) -> None:
        value = {
            "example": Example("тест", (1, 2), Path("data/input.txt")),
            7: (Path("one"), Path("two")),
        }

        self.assertEqual(
            to_plain_data(value),
            {
                "example": {
                    "name": "тест",
                    "values": [1, 2],
                    "path": "data/input.txt",
                },
                "7": ["one", "two"],
            },
        )

    def test_dump_creates_parents_and_round_trips_safe_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/result.yaml"
            value = Example("распределение", (10, 20), Path("matrix.npz"))

            dump_yaml(path, value)

            self.assertEqual(
                load_yaml(path),
                {
                    "name": "распределение",
                    "values": [10, 20],
                    "path": "matrix.npz",
                },
            )
            self.assertIn("распределение", path.read_text(encoding="utf-8"))

    def test_no_overwrite_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.yaml"
            path.write_text("old: value\n", encoding="utf-8")

            with self.assertRaisesRegex(YamlError, "уже существует"):
                dump_yaml(path, {"new": "value"}, overwrite=False)

            self.assertEqual(path.read_text(encoding="utf-8"), "old: value\n")

    def test_create_parents_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing/result.yaml"

            with self.assertRaisesRegex(YamlError, "Не удалось записать YAML"):
                dump_yaml(path, {"value": 1}, create_parents=False)

            self.assertFalse(path.exists())

    def test_unsupported_value_does_not_truncate_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.yaml"
            path.write_text("old: value\n", encoding="utf-8")

            with self.assertRaisesRegex(YamlError, "представить данные"):
                dump_yaml(path, {"unsupported": object()})

            self.assertEqual(path.read_text(encoding="utf-8"), "old: value\n")


if __name__ == "__main__":
    unittest.main()
