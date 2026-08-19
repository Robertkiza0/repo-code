"""User model and a tiny in-memory repository."""
from typing import Dict, List, Optional


class User:
    """Represents an application user."""

    def __init__(self, user_id: int, first_name: str, last_name: str, email: str):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.email = email

    @property
    def full_name(self) -> str:
        """Return the user's full name."""
        return f"{self.first_name} {self.last_name}"

    def to_dict(self) -> Dict:
        """Serialize the user to a plain dict."""
        return {
            "user_id": self.user_id,
            "full_name": self.full_name,
            "email": self.email,
        }


class UserRepository:
    """A minimal in-memory store for User objects."""

    def __init__(self):
        self._users: Dict[int, User] = {}

    def add(self, user: User) -> None:
        """Add or replace a user by id."""
        self._users[user.user_id] = user

    def get(self, user_id: int) -> Optional[User]:
        """Look up a user by id, or None if missing."""
        return self._users.get(user_id)

    def all(self) -> List[User]:
        """Return every stored user."""
        return list(self._users.values())
