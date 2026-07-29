from __future__ import annotations

from pathlib import Path
import sys
import unittest


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from schema_validation import SchemaValidator


class ExampleSchemaError(ValueError):
    pass


class SchemaValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = SchemaValidator(ExampleSchemaError)

    def test_uses_selected_exception_type(self) -> None:
        with self.assertRaises(ExampleSchemaError):
            self.validator.mapping([], "root")

    def test_error_type_must_be_exception_class(self) -> None:
        for value in (ValueError("instance"), str, 1):
            with self.subTest(value=value), self.assertRaisesRegex(
                TypeError, "классом исключения"
            ):
                SchemaValidator(value)  # type: ignore[arg-type]

    def test_mapping_requires_dictionary_with_string_keys(self) -> None:
        with self.assertRaisesRegex(ExampleSchemaError, "ожидался объект"):
            self.validator.mapping([], "root")
        with self.assertRaisesRegex(ExampleSchemaError, "должны быть строками"):
            self.validator.mapping({1: "value"}, "root")
        value = {"field": 1}
        self.assertIs(self.validator.mapping(value, "root"), value)

    def test_fields_reject_unknown_and_missing_names(self) -> None:
        with self.assertRaisesRegex(ExampleSchemaError, "неизвестные поля: extra"):
            self.validator.fields(
                {"required": 1, "extra": 2},
                "root",
                required={"required"},
            )
        with self.assertRaisesRegex(
            ExampleSchemaError, "отсутствуют обязательные поля: first, second"
        ):
            self.validator.fields(
                {},
                "root",
                required={"second", "first"},
            )

    def test_fields_return_validated_mapping(self) -> None:
        value = {"required": 1, "optional": 2}
        self.assertIs(
            self.validator.fields(
                value,
                "root",
                required={"required"},
                optional={"optional"},
            ),
            value,
        )

    def test_invalid_field_declaration_is_programming_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "одновременно"):
            self.validator.fields(
                {}, "root", required={"same"}, optional={"same"}
            )
        with self.assertRaisesRegex(TypeError, "Имена полей"):
            self.validator.fields({}, "root", required={1})  # type: ignore[arg-type]

    def test_string_rejects_wrong_type_and_empty_value(self) -> None:
        with self.assertRaisesRegex(ExampleSchemaError, "ожидалась строка"):
            self.validator.string(1, "field")
        with self.assertRaisesRegex(ExampleSchemaError, "не должна быть пустой"):
            self.validator.string("", "field")
        self.assertEqual(
            self.validator.string("", "field", allow_empty=True),
            "",
        )

    def test_boolean_is_strict(self) -> None:
        self.assertTrue(self.validator.boolean(True, "field"))
        for value in (1, 0, "true", None):
            with self.subTest(value=value), self.assertRaisesRegex(
                ExampleSchemaError, "логическое значение"
            ):
                self.validator.boolean(value, "field")

    def test_integer_rejects_boolean_and_checks_bounds(self) -> None:
        self.assertEqual(self.validator.integer(2, "field"), 2)
        with self.assertRaisesRegex(ExampleSchemaError, "целое число"):
            self.validator.integer(True, "field")
        with self.assertRaisesRegex(ExampleSchemaError, "целое число"):
            self.validator.integer(1.0, "field")
        with self.assertRaisesRegex(ExampleSchemaError, "положительным"):
            self.validator.integer(0, "field", positive=True)
        self.assertEqual(self.validator.integer(0, "field", non_negative=True), 0)
        with self.assertRaisesRegex(ExampleSchemaError, "не должно быть отрицательным"):
            self.validator.integer(-1, "field", non_negative=True)

    def test_number_accepts_integer_and_float_but_rejects_boolean(self) -> None:
        self.assertEqual(self.validator.number(2, "field"), 2.0)
        self.assertEqual(self.validator.number(2.5, "field"), 2.5)
        with self.assertRaisesRegex(ExampleSchemaError, "ожидалось число"):
            self.validator.number(True, "field")

    def test_number_rejects_non_finite_values(self) -> None:
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(
                ExampleSchemaError, "конечным"
            ):
                self.validator.number(value, "field")

    def test_number_checks_bounds(self) -> None:
        with self.assertRaisesRegex(ExampleSchemaError, "положительным"):
            self.validator.number(0, "field", positive=True)
        with self.assertRaisesRegex(ExampleSchemaError, "не должно быть отрицательным"):
            self.validator.number(-0.1, "field", non_negative=True)
        self.assertEqual(
            self.validator.number(0, "field", non_negative=True),
            0.0,
        )

    def test_choice_is_strict_and_reports_sorted_values(self) -> None:
        self.assertEqual(
            self.validator.choice("old", "position", {"new", "old"}),
            "old",
        )
        with self.assertRaisesRegex(
            ExampleSchemaError, "допустимо: new, old"
        ):
            self.validator.choice("other", "position", {"old", "new"})
        with self.assertRaisesRegex(ValueError, "непустым набором строк"):
            self.validator.choice("value", "field", set())

    def test_version_requires_exact_integer(self) -> None:
        self.assertEqual(
            self.validator.version(1, "schema_version", supported=1),
            1,
        )
        with self.assertRaisesRegex(ExampleSchemaError, "ожидалось целое число"):
            self.validator.version(True, "schema_version", supported=1)
        with self.assertRaisesRegex(ExampleSchemaError, "версия 2 не поддерживается"):
            self.validator.version(2, "schema_version", supported=1)
        with self.assertRaisesRegex(TypeError, "целым числом"):
            self.validator.version(1, "schema_version", supported=True)


if __name__ == "__main__":
    unittest.main()
