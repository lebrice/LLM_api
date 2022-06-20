""" API for querying a large language model. """
from __future__ import annotations
import logging

from fastapi import FastAPI
import numpy
from pydantic import BaseModel, Field
from torch import Tensor
from model import (
    DEFAULT_CAPACITY,
    Capacity,
    get_hidden_state,
    get_response_text,
    load_completion_model,
    load_embedding_model,
)
from dataclasses import dataclass
import socket
import os
from fastapi import Request
from logging import getLogger as get_logger

# TODO: Setup logging correctly with FastAPI.
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


def write_server_address_to_file():
    node_hostname = socket.gethostname()
    # TODO: We could perhaps detect the server port that is being used by FASTAPI programmatically?
    port = int(os.environ.get("SERVER_PORT", "8000"))

    with open("server.txt", "w") as f:
        address_string = f"{node_hostname}:{port}"
        print(f"Writing {address_string=} to server.txt")
        f.write(address_string)


app = FastAPI(on_startup=[write_server_address_to_file])


@app.get("/")
def root():
    return {"Hello": "World"}


@app.post("/load_model")
def load_model(capacity: Capacity = DEFAULT_CAPACITY):
    """Loads the OPT model with the given capacity, so that the other endpoints can use it."""
    print(f"Loading model with {capacity} parameters...")
    load_completion_model(capacity=capacity)
    load_embedding_model(capacity=capacity)
    print(f"Done loading model with capacity of {capacity} parameters.")
    return {"status": "ok"}


@dataclass
class CompletionResponse:
    prompt: str
    response: str


@app.get("/complete/")
def get_completion(
    *, prompt: str, max_response_length: int = 30, capacity: Capacity = DEFAULT_CAPACITY
) -> CompletionResponse:
    response_text: str = get_response_text(
        prompt, max_length=max_response_length, capacity=capacity
    )
    return CompletionResponse(
        prompt=prompt,
        response=response_text,
    )


@dataclass
class EmbeddingResponse:
    prompt: str
    hidden_state: list[float]


@app.get("/embedding/")
def get_embedding(
    *, prompt: str, capacity: Capacity = DEFAULT_CAPACITY
) -> EmbeddingResponse:
    """Gets the embedding vector for a given prompt."""
    hidden_state = get_hidden_state(prompt, capacity=capacity)
    return EmbeddingResponse(
        prompt=prompt, hidden_state=hidden_state.flatten().cpu().tolist()
    )
