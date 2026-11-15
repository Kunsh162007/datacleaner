"""Test doubles for the LLM client."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


class FakeClient:
    """Returns queued responses and records every prompt it receives."""

    model = "fake-model"

    def __init__(self, *responses: str | Mapping[str, Any] | Exception) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, str, Mapping[str, Any]]] = []

    def complete_json(self, system: str, user: str, response_schema: Mapping[str, Any]) -> str:
        self.calls.append((system, user, response_schema))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)


def action(id_: str, column: str, name: str, params: dict | None = None) -> dict:
    return {"id": id_, "column": column, "action": name, "params": params or {}, "reason": "t"}
