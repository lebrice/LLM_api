from pathlib import Path
from pydantic import BaseSettings
import uvicorn

from simple_parsing import ArgumentParser, field
from dataclasses import dataclass
import os
from app.server import ServerConfig, app, get_settings


def main():
    parser = ArgumentParser(description="Run the server.")
    parser.add_arguments(ServerConfig, "server", default=ServerConfig())
    args = parser.parse_args()
    server_config: ServerConfig = args.server

    print(f"Running the server with the following settings: {server_config}")

    app.dependency_overrides[get_settings] = lambda: server_config

    # global DEFAULT_CAPACITY
    # DEFAULT_CAPACITY = server_config.model_capacity
    # os.environ["MODEL_CAPACITY"] = server_config.model_capacity
    # os.environ["SERVER_PORT"] = str(server_config.port)
    uvicorn.run(
        app,
        # "server:app",
        port=server_config.port,
        log_level="debug",
        # reload=server_config.reload,
    )


if __name__ == "__main__":
    main()
