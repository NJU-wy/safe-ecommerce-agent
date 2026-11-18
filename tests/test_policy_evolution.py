import sqlite3

import pytest

from app.agent.strategies.policy import (
    current_policy, passes_gate, policy_prompt, save_validated, switch_policy,
)
from app.config.settings import settings


@pytest.fixture
def policy_store(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "policy_db_path", str(tmp_path / "policies.sqlite3"))
    return tmp_path / "policies.sqlite3"


def _evidence(improved=None, regressed=None, errors=0):
    return {
        "development_cases": ["failed-case"],
        "paired": {"improved": improved or [], "regressed": regressed or []},
        "errors": errors,
    }


def test_policy_is_empty_by_default(policy_store):
    assert current_policy("single") == (0, ())
    assert policy_prompt("single") == ""


def test_gate_requires_improvement_without_regression():
    assert passes_gate({"improved": ["a"], "regressed": []}, 0)
    assert not passes_gate({"improved": [], "regressed": []}, 0)
    assert not passes_gate({"improved": ["a"], "regressed": ["b"]}, 0)
    assert not passes_gate({"improved": ["a"], "regressed": []}, 1)


def test_validated_policy_requires_explicit_adoption(policy_store):
    version = save_validated("single", 0, ("lookup_first",), _evidence(["a"]))
    assert current_policy("single") == (0, ())
    switch_policy("single", version)
    assert current_policy("single") == (version, ("lookup_first",))
    assert "先用可用工具核对" in policy_prompt("single")


def test_policy_can_rollback(policy_store):
    first = save_validated("single", 0, ("lookup_first",), _evidence(["a"]))
    switch_policy("single", first)
    second = save_validated(
        "single", first, ("lookup_first", "ground_reply"), _evidence(["b"]),
    )
    switch_policy("single", second)
    switch_policy("single", None, rollback_from=second)
    assert current_policy("single") == (first, ("lookup_first",))


def test_rejects_unvalidated_or_stale_policy(policy_store):
    with pytest.raises(ValueError):
        save_validated("single", 0, ("lookup_first",), _evidence())
    version = save_validated("single", 0, ("lookup_first",), _evidence(["a"]))
    switch_policy("single", version)
    with pytest.raises(ValueError):
        save_validated("single", 0, ("ground_reply",), _evidence(["b"]))
    with sqlite3.connect(policy_store) as db:
        assert db.execute("SELECT COUNT(*) FROM versions").fetchone()[0] == 1
