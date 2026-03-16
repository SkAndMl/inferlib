import argparse
import os
import shutil
import subprocess
import uvicorn

from pathlib import Path

from inferlib.core import SUPPORTED_MODEL_LIST


def _ensure_frontend_bundle() -> None:
    server_static_index = Path(__file__).resolve().parent / "static" / "index.html"
    frontend_dir = Path(__file__).resolve().parents[2] / "frontend"
    frontend_dist_index = frontend_dir / "dist" / "index.html"

    if frontend_dir.exists():
        if frontend_dist_index.exists():
            return

        npm_bin = shutil.which("npm")
        if npm_bin is None:
            raise RuntimeError(
                "Frontend build is missing and npm is not installed. "
                "Install npm or provide compiled assets."
            )

        print("Frontend bundle not found. Building UI...")

        if not (frontend_dir / "node_modules").exists():
            subprocess.run([npm_bin, "install"], cwd=frontend_dir, check=True)

        subprocess.run([npm_bin, "run", "build"], cwd=frontend_dir, check=True)

        if not frontend_dist_index.exists():
            raise RuntimeError("Frontend build finished without producing dist/index.html.")
        return

    if server_static_index.exists():
        return

    raise RuntimeError("No frontend assets found (missing frontend/dist and server/static).")


def main():
    parser = argparse.ArgumentParser(prog="inferlib")
    subparser = parser.add_subparsers(dest="command")
    serve = subparser.add_parser(name="serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--model-class",
        type=str,
        default="Qwen/Qwen3-0.6B",
        choices=SUPPORTED_MODEL_LIST,
    )

    args = parser.parse_args()
    if args.command == "serve":
        os.environ["INFERLIB_MODEL_CLASS"] = args.model_class
        try:
            _ensure_frontend_bundle()
        except (RuntimeError, subprocess.CalledProcessError) as error:
            raise SystemExit(f"Failed to prepare frontend bundle: {error}") from error
        uvicorn.run("inferlib.server.app:app", host=args.host, port=args.port)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
