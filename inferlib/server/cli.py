from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from pathlib import Path

import uvicorn

from inferlib.logging import configure_logging, ensure_logging_configured
from inferlib.server.app import create_app
from inferlib.server.log import logger
from inferlib.settings import load_settings


def frontend_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "frontend"


def frontend_dist_index() -> Path:
    return frontend_dir() / "dist" / "index.html"


def server_static_index() -> Path:
    return Path(__file__).resolve().parent / "static" / "index.html"


def ensure_frontend_bundle_available() -> None:
    if frontend_dist_index().exists() or server_static_index().exists():
        return
    raise RuntimeError(
        "Frontend bundle not found. Run `inferlib build-frontend` or start with `inferlib serve --no-ui`."
    )


def build_frontend_bundle() -> None:
    ensure_logging_configured()
    root = frontend_dir()
    if not root.exists():
        raise RuntimeError("frontend directory not found in this checkout")

    npm_bin = shutil.which("npm")
    if npm_bin is None:
        raise RuntimeError("npm is not installed")

    install_command = "ci" if (root / "package-lock.json").exists() else "install"
    if not (root / "node_modules").exists():
        logger.info("installing frontend dependencies", extra={"command": f"npm {install_command}"})
        subprocess.run([npm_bin, install_command], cwd=root, check=True)

    logger.info("building frontend bundle", extra={"frontend_dir": str(root)})
    subprocess.run([npm_bin, "run", "build"], cwd=root, check=True)

    if not frontend_dist_index().exists():
        raise RuntimeError("frontend build finished without producing dist/index.html")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inferlib")
    subparser = parser.add_subparsers(dest="command", required=True)

    serve = subparser.add_parser(name="serve")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--model-class")
    serve.add_argument("--db-path")
    serve.add_argument("--page-size", type=int)
    serve.add_argument("--batch-size", type=int)
    serve.add_argument("--max-active-sequences", type=int)
    serve.add_argument("--memory-limit-bytes", type=int)
    serve.add_argument("--log-level")
    serve.add_argument("--log-format", choices=["text", "json"])
    serve.add_argument("--no-ui", action="store_true")

    subparser.add_parser(name="build-frontend")
    return parser


def settings_overrides_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "host": args.host,
        "port": args.port,
        "model_class": args.model_class,
        "db_path": args.db_path,
        "page_size": args.page_size,
        "batch_size": args.batch_size,
        "max_active_sequences": args.max_active_sequences,
        "memory_limit_bytes": args.memory_limit_bytes,
        "log_level": args.log_level,
        "log_format": args.log_format,
        "serve_ui": False if args.no_ui else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-frontend":
        try:
            build_frontend_bundle()
        except (RuntimeError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"Failed to build frontend: {error}") from error
        return 0

    settings = load_settings(overrides=settings_overrides_from_args(args))
    configure_logging(level=settings.log_level, fmt=settings.log_format)

    if settings.serve_ui:
        try:
            ensure_frontend_bundle_available()
        except RuntimeError as error:
            raise SystemExit(str(error)) from error

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
