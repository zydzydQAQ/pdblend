"""Explicit configuration and lifecycle entry for the modular serving runtime."""
import argparse
import json
from pathlib import Path
from aiohttp import web
from .runtime import Controller, STRATEGIES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    parser.add_argument("--strategy",choices=STRATEGIES)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    if args.strategy:
        config["strategy"]=args.strategy
    web.run_app(Controller(config).application(),host="127.0.0.1",port=config.get("port",8000),access_log=None)


if __name__=="__main__":
    main()
