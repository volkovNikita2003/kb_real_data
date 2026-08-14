"""Shared, schema-agnostic helpers for safe YAML input and output."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from yaml.constructor import ConstructorError


YAML_OMIT_NONE = "yaml_omit_none"


class YamlError(ValueError):
    """A YAML document cannot be read, represented, or written safely."""


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """PyYAML safe loader which additionally rejects duplicate keys."""


def _construct_unique_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_yaml(path: str | Path) -> Any:
    """Load one safe, non-empty YAML document and reject duplicate keys."""
    source = Path(path)
    if not source.exists():
        raise YamlError(f"Файл YAML не найден: {source}")
    if not source.is_file():
        raise YamlError(f"Путь YAML не является файлом: {source}")
    try:
        with source.open("r", encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=_UniqueKeySafeLoader)
    except (OSError, yaml.YAMLError) as error:
        raise YamlError(f"Не удалось прочитать YAML {source}: {error}") from error
    if value is None:
        raise YamlError(f"Файл YAML пуст: {source}")
    return value


def to_plain_data(value: Any) -> Any:
    """Recursively convert common project objects to safe YAML containers."""
    if is_dataclass(value) and not isinstance(value, type):
        result: dict[str, Any] = {}
        for item in fields(value):
            item_value = getattr(value, item.name)
            if item_value is None and item.metadata.get(YAML_OMIT_NONE, False):
                continue
            result[item.name] = to_plain_data(item_value)
        return result
    if isinstance(value, Mapping):
        return {
            str(key): to_plain_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def dump_yaml(
    path: str | Path,
    value: Any,
    *,
    create_parents: bool = True,
    overwrite: bool = True,
) -> None:
    """Serialize *value* as safe YAML and write it using UTF-8.

    Serialization is completed before the destination is opened, so an
    unsupported value cannot truncate an existing file.
    """
    destination = Path(path)
    try:
        document = yaml.safe_dump(
            to_plain_data(value),
            allow_unicode=True,
            sort_keys=False,
        )
    except yaml.YAMLError as error:
        raise YamlError(
            f"Не удалось представить данные в формате YAML для {destination}: {error}"
        ) from error

    try:
        if create_parents:
            destination.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if overwrite else "x"
        with destination.open(mode, encoding="utf-8") as stream:
            stream.write(document)
    except FileExistsError as error:
        raise YamlError(f"Файл YAML уже существует: {destination}") from error
    except OSError as error:
        raise YamlError(f"Не удалось записать YAML {destination}: {error}") from error
