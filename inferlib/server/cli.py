import argparse
import os
import uvicorn

from inferlib.core import SUPPORTED_MODEL_LIST


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
        uvicorn.run("inferlib.server.app:app", host=args.host, port=args.port)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
