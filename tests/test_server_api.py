from __future__ import annotations

from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from inferlib.server.app import create_app
from inferlib.settings import load_settings
from tests.fakes import FakeServerEngine


def _make_client(tmp_path: Path) -> TestClient:
    settings = load_settings(
        overrides={
            "model_class": "fake-model",
            "db_path": str(tmp_path / "chats.sqlite3"),
            "serve_ui": False,
        }
    )
    app = create_app(settings, engine_factory=FakeServerEngine)
    return TestClient(app)


def test_health_and_models_endpoints(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        assert client.get("/health").json() == {"status": "ok", "model": "fake-model"}
        assert client.get("/v1/models").json() == {
            "object": "list",
            "data": [{"id": "fake-model", "object": "model", "owned_by": "inferlib"}],
        }


def test_chat_completion_non_streaming(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
                "max_tokens": 4,
            },
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["model"] == "fake-model"
    assert payload["choices"][0]["message"]["content"] == "hello world"
    assert payload["usage"]["completion_tokens"] == 2


def test_chat_completion_streaming(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 4,
            },
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert 'data: {"id":"' in body
    assert "data: [DONE]" in body


def test_chat_completion_rejects_unknown_model(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "other-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_not_loaded"


def test_chat_completion_rejects_oversize_prompt(tmp_path: Path) -> None:
    with _make_client(tmp_path) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "fake-model",
                "messages": [{"role": "user", "content": "x" * 60}],
                "stream": False,
                "max_tokens": 10,
            },
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"


def test_chat_crud_routes_use_error_envelope(tmp_path: Path) -> None:
    chat_id = "chat-1"
    with _make_client(tmp_path) as client:
        save_response = client.post(
            f"/v1/chats/{chat_id}/messages",
            json={"role": "user", "content": "hello there"},
        )
        assert save_response.status_code == 200

        list_response = client.get("/v1/chats")
        assert list_response.json()["chats"][0]["chat_id"] == chat_id

        messages_response = client.get(f"/v1/chats/{chat_id}/messages")
        assert messages_response.status_code == 200
        assert messages_response.json()["messages"][0]["content"] == "hello there"

        patch_response = client.patch(f"/v1/chats/{chat_id}", json={"title": "Renamed"})
        assert patch_response.status_code == 200
        assert patch_response.json()["title"] == "Renamed"

        delete_response = client.delete(f"/v1/chats/{chat_id}")
        assert delete_response.status_code == 200

        not_found_response = client.get(f"/v1/chats/{chat_id}")
        assert not_found_response.status_code == 404
        assert not_found_response.json()["error"]["code"] == "chat_not_found"


def test_startup_rejects_directory_db_path(tmp_path: Path) -> None:
    db_dir = tmp_path / "chats.db"
    db_dir.mkdir()
    settings = load_settings(
        overrides={
            "model_class": "fake-model",
            "db_path": str(db_dir),
            "serve_ui": False,
        }
    )
    app = create_app(settings, engine_factory=FakeServerEngine)

    with pytest.raises(RuntimeError, match="Database path points to a directory"):
        with TestClient(app):
            pass
