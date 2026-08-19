"""Simple calculator module."""
from typing import List


class Calculator:
    """A basic calculator that keeps a running history."""

    def __init__(self):
        self.history: List[str] = []

    def add(self, a: float, b: float) -> float:
        """Add two numbers and record the operation."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result

    def subtract(self, a: float, b: float) -> float:
        """Subtract b from a and record the operation."""
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result

    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers and record the operation."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result

    def divide(self, a: float, b: float) -> float:
        """Divide a by b and record the operation."""
        if b == 0:
            raise ZeroDivisionError("cannot divide by zero")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result

    def clear_history(self) -> None:
        """Clear the recorded operation history."""
        self.history.clear()


def average(values: List[float]) -> float:
    """Return the arithmetic mean of values."""
    return sum(values) / len(values)
