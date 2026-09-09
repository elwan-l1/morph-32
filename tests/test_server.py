# ruff: noqa: E402
"""Request routing and template safeguards without loading model weights."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("mlx.core")
from morph32 import server


def test_provider_rejects_other_models_before_loading(monkeypatch, tmp_path):
    provider = object.__new__(server.MorphModelProvider)
    provider.checkpoint = (tmp_path / "selected.morph").resolve()
    provider.model_id = "selected"
    monkeypatch.setattr(server, "load", lambda *a, **k: pytest.fail("Unexpected model load"))
    with pytest.raises(ValueError, match="only provides"):
        provider._load(str(tmp_path / "another.morph"))
    with pytest.raises(ValueError, match="Adapters and draft"):
        provider._load(str(provider.checkpoint), adapter_path="adapter")


def test_non_thinking_and_context_budget(monkeypatch):
    @dataclass
    class Arguments:
        max_tokens: int
        chat_template_kwargs: dict

    def tokenize(self, tokenizer, request, args):
        assert args.chat_template_kwargs == {"enable_thinking": False, "custom": "kept"}
        return list(range(12)), [], [], "normal"

    monkeypatch.setattr(server.ResponseGenerator, "_tokenize", tokenize)
    generator = object.__new__(server.MorphResponseGenerator)
    generator.model_provider = SimpleNamespace(cli_args=SimpleNamespace(context_length=16))
    args = Arguments(4, {"enable_thinking": True, "custom": "kept"})
    assert len(generator._tokenize(None, None, args)[0]) == 12
    assert args.chat_template_kwargs["enable_thinking"] is True
    with pytest.raises(ValueError, match="exceeds 16"):
        generator._tokenize(None, None, Arguments(5, args.chat_template_kwargs))


def test_example_matches_server_defaults():
    import json

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "docs/opencode.example.json").read_text())
    provider = config["provider"]["morph32-local"]
    assert config["model"] == "morph32-local/" + server.DEFAULT_MODEL_ID
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8081/v1"
    assert provider["models"][server.DEFAULT_MODEL_ID]["limit"] == {
        "context": 32768,
        "output": 4096,
    }
