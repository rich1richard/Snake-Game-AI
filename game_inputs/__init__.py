from enum import Enum
from typing import Any, Sequence, Iterable


class Direction(Enum):
    RIGHT = 0
    UP = 1
    LEFT = 2
    DOWN = 3


class Game:
    def reset(self) -> tuple[Iterable[Sequence[int]], dict[str, Any]]:
        raise NotImplementedError

    def step(self) -> tuple[bool, int]:
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def must_quit(self) -> bool:
        raise NotImplementedError

    def score(self) -> int:
        raise NotImplementedError

    def snake_length(self) -> int:
        raise NotImplementedError

    def state_size(self) -> int:
        raise NotImplementedError


class BaseAgent:
    def __init__(self, game):
        self._game = game
        self._stop_game = False
        self._direction = Direction.RIGHT

    def update(self):
        raise NotImplementedError

    def stop_game(self):
        self._stop_game = True

    def set_direction(self, direction: Direction):
        self._direction = direction

    def must_stop_game(self) -> bool:
        return self._stop_game

    def game(self) -> Game:
        return self._game
