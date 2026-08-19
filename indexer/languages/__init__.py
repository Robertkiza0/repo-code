"""Language registry.

Adding a new language means: write a BaseExtractor subclass and register a
LanguageConfig for it here (or in its own registration call) -- nothing in
repo_parser.py needs to change.
"""
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from tree_sitter import Language

from indexer.languages.base import BaseExtractor
from indexer.languages.python_extractor import PythonExtractor

import tree_sitter_python as ts_python


@dataclass(frozen=True)
class LanguageConfig:
    name: str
    extensions: Tuple[str, ...]
    language_factory: Callable[[], Language]
    extractor: BaseExtractor


LANGUAGES: Dict[str, LanguageConfig] = {}
_EXTENSION_TO_LANGUAGE: Dict[str, str] = {}


def register_language(config: LanguageConfig) -> None:
    LANGUAGES[config.name] = config
    for ext in config.extensions:
        _EXTENSION_TO_LANGUAGE[ext] = config.name


def get_language_for_extension(extension: str) -> "LanguageConfig | None":
    name = _EXTENSION_TO_LANGUAGE.get(extension)
    return LANGUAGES.get(name) if name else None


register_language(
    LanguageConfig(
        name="python",
        extensions=(".py",),
        language_factory=lambda: Language(ts_python.language()),
        extractor=PythonExtractor(),
    )
)
