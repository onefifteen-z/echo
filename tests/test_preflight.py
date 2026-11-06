from __future__ import annotations

import sys

import pytest

from echo import preflight


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.delenv("ECHO_SKIP_PREFLIGHT", raising=False)
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(preflight, "warm_imports", lambda modules: None)
    calls: list[tuple[str, ...]] = []
    result = {"ok": True}

    def fake_check(modules):
        calls.append(modules)
        return (True, "ok") if result["ok"] else (False, "Importing `x` exited with code 1.")

    monkeypatch.setattr(preflight, "check_imports_in_child", fake_check)
    return tmp_path, calls, result


def test_passing_check_is_remembered(env):
    prefix, calls, _ = env
    preflight.run_preflight(live=True)
    preflight.run_preflight(live=True)
    assert len(calls) == 1
    assert (prefix / ".echo-preflight-ok").is_file()


def test_changed_environment_runs_check_again(env, monkeypatch):
    _, calls, _ = env
    preflight.run_preflight(live=True)
    monkeypatch.setattr(preflight, "_preflight_key", lambda modules: "changed")
    preflight.run_preflight(live=True)
    assert len(calls) == 2


def test_failed_check_exits_and_is_not_remembered(env):
    prefix, calls, result = env
    result["ok"] = False
    with pytest.raises(SystemExit):
        preflight.run_preflight(live=True)
    assert not (prefix / ".echo-preflight-ok").exists()
    result["ok"] = True
    preflight.run_preflight(live=True)
    assert len(calls) == 2


def test_warm_flag(env, monkeypatch):
    warmed: list[tuple[str, ...]] = []
    monkeypatch.setattr(preflight, "warm_imports", warmed.append)
    preflight.run_preflight(live=False, warm=False)
    assert warmed == []
    preflight.run_preflight(live=False)
    assert warmed == [preflight.LAZY_NATIVE_MODULES]
