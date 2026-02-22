import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(prog="inferlib")
    subparser = parser.add_subparsers(dest="command")
    serve = subparser.add_parser(name="serve")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8000)

    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run("inferlib.server.app:app", host=args.host, port=args.port)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
