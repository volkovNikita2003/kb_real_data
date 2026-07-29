"""Reusable primitives for strict validation of decoded configuration data."""

from __future__ import annotations

from collections.abc import Collection
import math
from typing import Any, NoReturn


class SchemaValidator:
    """Validate plain decoded values and raise a caller-selected error type.

    The validator is deliberately independent of YAML and concrete project
    schemas.  It can therefore be reused for input parameters and for result
    documents while preserving their public exception types.
    """

    def __init__(self, error_type: type[Exception]) -> None:
        if not isinstance(error_type, type) or not issubclass(error_type, Exception):
            raise TypeError("error_type должен быть классом исключения")
        self.error_type = error_type

    def _fail(self, message: str) -> NoReturn:
        raise self.error_type(message)

    def mapping(self, value: Any, path: str) -> dict[str, Any]:
        if not isinstance(value, dict):
            self._fail(f"{path}: ожидался объект (mapping)")
        if not all(isinstance(key, str) for key in value):
            self._fail(f"{path}: имена всех полей должны быть строками")
        return value

    def fields(
        self,
        value: Any,
        path: str,
        *,
        required: Collection[str] = frozenset(),
        optional: Collection[str] = frozenset(),
    ) -> dict[str, Any]:
        required_fields = frozenset(required)
        optional_fields = frozenset(optional)
        if not all(isinstance(name, str) for name in required_fields | optional_fields):
            raise TypeError("Имена полей схемы должны быть строками")
        overlap = required_fields & optional_fields
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(
                f"Поля не могут быть одновременно обязательными и необязательными: {names}"
            )

        data = self.mapping(value, path)
        unknown = sorted(data.keys() - required_fields - optional_fields)
        if unknown:
            self._fail(f"{path}: неизвестные поля: {', '.join(unknown)}")
        missing = sorted(required_fields - data.keys())
        if missing:
            self._fail(
                f"{path}: отсутствуют обязательные поля: {', '.join(missing)}"
            )
        return data

    def string(self, value: Any, path: str, *, allow_empty: bool = False) -> str:
        if not isinstance(value, str):
            self._fail(f"{path}: ожидалась строка")
        if not allow_empty and not value:
            self._fail(f"{path}: строка не должна быть пустой")
        return value

    def boolean(self, value: Any, path: str) -> bool:
        if type(value) is not bool:
            self._fail(f"{path}: ожидалось логическое значение")
        return value

    def integer(
        self,
        value: Any,
        path: str,
        *,
        positive: bool = False,
        non_negative: bool = False,
    ) -> int:
        if type(value) is not int:
            self._fail(f"{path}: ожидалось целое число")
        if positive and value <= 0:
            self._fail(f"{path}: число должно быть положительным")
        if non_negative and value < 0:
            self._fail(f"{path}: число не должно быть отрицательным")
        return value

    def number(
        self,
        value: Any,
        path: str,
        *,
        positive: bool = False,
        non_negative: bool = False,
    ) -> float:
        if type(value) not in (int, float):
            self._fail(f"{path}: ожидалось число")
        result = float(value)
        if not math.isfinite(result):
            self._fail(f"{path}: число должно быть конечным")
        if positive and result <= 0:
            self._fail(f"{path}: число должно быть положительным")
        if non_negative and result < 0:
            self._fail(f"{path}: число не должно быть отрицательным")
        return result

    def choice(
        self,
        value: Any,
        path: str,
        choices: Collection[str],
    ) -> str:
        allowed_values = frozenset(choices)
        if not allowed_values or not all(
            isinstance(item, str) for item in allowed_values
        ):
            raise ValueError("Допустимые значения должны быть непустым набором строк")
        result = self.string(value, path)
        if result not in allowed_values:
            allowed = ", ".join(sorted(allowed_values))
            self._fail(
                f"{path}: неизвестное значение {result!r}; допустимо: {allowed}"
            )
        return result

    def version(self, value: Any, path: str, *, supported: int) -> int:
        if type(supported) is not int:
            raise TypeError("Поддерживаемая версия должна быть целым числом")
        result = self.integer(value, path)
        if result != supported:
            self._fail(
                f"{path}: версия {result} не поддерживается; "
                f"поддерживается версия {supported}"
            )
        return result
