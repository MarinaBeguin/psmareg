"""Conversion between sectioned configurations and flat parameter mappings.

A configuration is a frozen dataclass whose fields are themselves frozen
dataclasses ("sections"). Parameter names are unique across sections, so a
configuration can also be addressed by a flat ``name -> value`` mapping, which
is the form used by recipes, command-line overrides and checkpoint metadata.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import fields
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Collection, Mapping, TypeVar

ConfigT = TypeVar("ConfigT")


def coerce_cli_override(
    name: str,
    raw: str,
    current: Any,
    *,
    path_fields: Collection[str],
    nullable_fields: Collection[str],
    optional_numeric_fields: Mapping[str, type] | None = None,
) -> Any:
    """Convert a ``--set NAME=VALUE`` string for either training pipeline.

    ``none`` and ``null`` both disable declared optional parameters. Lists for
    tuple parameters use JSON syntax, and booleans accept true/false or 1/0.
    Field collections make nullability explicit even when a recipe supplies a
    string path rather than a ``Path`` instance.

    Args:
        name: Parameter being overridden.
        raw: Text given on the command line.
        current: Current value, whose type selects the conversion.
        path_fields: Parameters converted to ``Path``.
        nullable_fields: Parameters that ``none`` or ``null`` may disable.
        optional_numeric_fields: Nullable numeric parameters and their types.

    Returns:
        Converted value.

    Raises:
        ValueError: If the text cannot disable the parameter or does not match
            its type.
        TypeError: If the current value has no supported conversion.
    """
    if raw.lower() in {"none", "null"}:
        if name in nullable_fields:
            return None
        raise ValueError(f"{name} does not accept None.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    if value is None:
        if name in nullable_fields:
            return None
        raise ValueError(f"{name} does not accept None.")
    if name in path_fields:
        if not isinstance(value, str):
            raise ValueError(f"{name} expects a path string.")
        return Path(value)
    if optional_numeric_fields and name in optional_numeric_fields:
        return optional_numeric_fields[name](value)
    if isinstance(current, tuple):
        if not isinstance(value, list):
            raise ValueError(
                "Tuple overrides must use a JSON list, e.g. [2,4,8].")
        return tuple(value)
    return _coerce_scalar(name, value, current)


def _coerce_scalar(name: str, value: Any, current: Any) -> Any:
    """Convert a decoded override to the scalar type of the current value."""
    if isinstance(current, bool):
        if isinstance(value,
                      str) and value.lower() in {"true", "false", "1", "0"}:
            return value.lower() in {"true", "1"}
        if isinstance(value, bool) or (value in (0, 1) and
                                       isinstance(value, int)):
            return bool(value)
        raise ValueError(f"{name} expects a boolean.")
    for scalar_type, label in ((int, "an integer"), (float, "a float")):
        if isinstance(current, scalar_type):
            if isinstance(value, bool):
                raise ValueError(f"{name} expects {label}.")
            return scalar_type(value)
    if isinstance(current, str):
        return str(value)
    raise TypeError(f"No override conversion for {name}.")


def check_fields(values: Mapping[str, Any], names: Collection[str],
                 is_valid: Callable[[Any], bool], requirement: str) -> None:
    """Require every named parameter to satisfy one range rule.

    Args:
        values: Flat configuration.
        names: Parameters the rule applies to, checked in order.
        is_valid: Predicate accepting a valid value.
        requirement: Rule completing the message ``"<name> <requirement>."``.

    Raises:
        ValueError: For the first parameter that violates the rule.
    """
    for name in names:
        if not is_valid(values[name]):
            raise ValueError(f"{name} {requirement}.")


def load_python_recipe(path: Path, *, module_name: str,
                       error_label: str) -> ModuleType:
    """Import a Python recipe without modifying the import search path.

    Args:
        path: Python file holding uppercase configuration constants.
        module_name: Name given to the imported module.
        error_label: Description of the recipe used in the error message.

    Returns:
        Imported recipe module.

    Raises:
        ValueError: If the file cannot be loaded as a module.
    """
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {error_label}: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_flat_overrides(
    values: dict[str, Any],
    overrides: dict[str, Any] | None,
    *,
    unknown_label: str,
    path_fields: Collection[str],
    nullable_fields: Collection[str],
    optional_numeric_fields: Mapping[str, type] | None = None,
) -> None:
    """Apply known flat overrides and normalize path parameters.

    Args:
        values: Flat configuration updated in place.
        overrides: Values supplied by the command line; strings are converted
            with ``coerce_cli_override``.
        unknown_label: Description used when a name is not a parameter.
        path_fields: Parameters converted to ``Path``.
        nullable_fields: Parameters that ``none`` or ``null`` may disable.
        optional_numeric_fields: Nullable numeric parameters and their types.

    Raises:
        ValueError: If an override names an unknown parameter.
    """
    for name, raw in (overrides or {}).items():
        if name not in values:
            raise ValueError(f"Unknown {unknown_label}: {name}")
        values[name] = coerce_cli_override(
            name,
            raw,
            values[name],
            path_fields=path_fields,
            nullable_fields=nullable_fields,
            optional_numeric_fields=optional_numeric_fields) if isinstance(
                raw, str) else raw
    for name in path_fields:
        if values[name] is not None and not isinstance(values[name], Path):
            values[name] = Path(values[name])


def flatten_config(config: Any) -> dict[str, Any]:
    """Flatten a sectioned configuration into parameter names.

    Args:
        config: Sectioned configuration instance.

    Returns:
        Mapping from parameter name to value, in declaration order.

    Raises:
        RuntimeError: If two sections declare the same parameter name.
    """
    values: dict[str, Any] = {}
    for section_field in fields(config):
        section = getattr(config, section_field.name)
        for item in fields(section):
            if item.name in values:
                raise RuntimeError(
                    f"Duplicate configuration parameter: {item.name}")
            values[item.name] = getattr(section, item.name)
    return values


def build_config(config_type: type[ConfigT], values: dict[str, Any]) -> ConfigT:
    """Build a sectioned configuration from flat parameter values.

    Args:
        config_type: Sectioned configuration class with default sections.
        values: Complete mapping from parameter name to value.

    Returns:
        Configuration whose sections hold the given values.
    """
    defaults: Any = config_type()
    sections = {}
    for section_field in fields(defaults):
        section_type = type(getattr(defaults, section_field.name))
        sections[section_field.name] = section_type(
            **{item.name: values[item.name] for item in fields(section_type)})
    return config_type(**sections)


def add_set_argument(parser: argparse.ArgumentParser, help_text: str) -> None:
    """Add the repeatable ``--set NAME=VALUE`` diagnostic override.

    Args:
        parser: Parser of a training entry point.
        help_text: Description shown by ``--help``.
    """
    parser.add_argument("--set",
                        action="append",
                        default=[],
                        metavar="NAME=VALUE",
                        help=help_text)


def named_overrides(cli: argparse.Namespace,
                    names: Collection[str]) -> dict[str, Any]:
    """Collect the dedicated command-line options that were provided."""
    return {
        name: getattr(cli, name)
        for name in names
        if getattr(cli, name) is not None
    }


def set_overrides(parser: argparse.ArgumentParser,
                  items: Collection[str]) -> dict[str, str]:
    """Split ``--set NAME=VALUE`` items into raw overrides.

    Args:
        parser: Parser used to report a malformed item.
        items: Values collected by the ``--set`` option, in order.

    Returns:
        Raw string value of each overridden parameter.
    """
    overrides: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            parser.error(f"--set expects NAME=VALUE, got {item!r}")
        name, raw = item.split("=", 1)
        overrides[name] = raw
    return overrides
