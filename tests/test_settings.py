from __future__ import annotations

from inferlib.server.cli import build_parser, settings_overrides_from_args
from inferlib.settings import load_settings


def test_load_settings_prefers_cli_overrides() -> None:
    settings = load_settings(
        overrides={
            "port": 9000,
            "serve_ui": False,
            "db_path": "/tmp/inferlib.sqlite3",
        },
        environ={
            "INFERLIB_PORT": "7000",
            "INFERLIB_UI": "true",
            "INFERLIB_DB_PATH": "/tmp/from-env.sqlite3",
        },
    )

    assert settings.port == 9000
    assert settings.serve_ui is False
    assert settings.db_path.name == "inferlib.sqlite3"


def test_load_settings_supports_legacy_db_path_env() -> None:
    settings = load_settings(environ={"DB_PATH": "/tmp/legacy.sqlite3"})

    assert settings.db_path.name == "legacy.sqlite3"


def test_cli_parser_exposes_new_serve_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--port",
            "8100",
            "--db-path",
            "/tmp/test.sqlite3",
            "--page-size",
            "16",
            "--log-format",
            "json",
            "--no-ui",
        ]
    )

    overrides = settings_overrides_from_args(args)

    assert overrides["port"] == 8100
    assert overrides["db_path"] == "/tmp/test.sqlite3"
    assert overrides["page_size"] == 16
    assert overrides["log_format"] == "json"
    assert overrides["serve_ui"] is False
