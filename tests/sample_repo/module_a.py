"""Module A: sample module for parser tests."""
import os
from typing import List, Optional


def greet(name: str, greeting: str = "Hello") -> str:
    """Return a greeting for name."""
    return f"{greeting}, {name}!"


class Greeter:
    """A simple greeter class."""

    def __init__(self, default_greeting: str = "Hi"):
        self.default_greeting = default_greeting

    def greet(self, name: str) -> str:
        """Greet name using the default greeting."""
        def _format(n: str) -> str:
            return n.strip().title()
        return f"{self.default_greeting}, {_format(name)}!"

    @staticmethod
    def shout(name: str) -> str:
        return f"HEY {name.upper()}!"
