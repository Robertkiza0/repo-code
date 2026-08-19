"""Module B: another sample module."""
from module_a import Greeter


class LoudGreeter(Greeter):
    def greet(self, name: str) -> str:
        return super().greet(name).upper()
