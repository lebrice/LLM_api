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
    load_tokenizer,
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

from fastapi.responses import RedirectResponse


@app.get("/")
def root(request: Request):
    return RedirectResponse(url=f"{request.base_url}docs")


@app.post("/load_model")
async def load_model(capacity: Capacity = DEFAULT_CAPACITY):
    """Loads the OPT model with the given capacity, so that the other endpoints can use it."""
    print(f"Loading model with {capacity} parameters...")
    unload_model()
    load_completion_model(capacity=capacity)
    load_embedding_model(capacity=capacity)
    load_tokenizer(capacity=capacity)
    global DEFAULT_CAPACITY
    DEFAULT_CAPACITY = capacity
    app.state.capacity = capacity
    print(f"Done loading model with capacity of {capacity} parameters.")
    return {"status": "ok"}


def unload_model():
    """Unloads the currently loaded model, to free the CUDA memory so another can be loaded."""
    print(f"Unloading all models...")
    load_completion_model.cache_clear()
    load_embedding_model.cache_clear()
    load_tokenizer.cache_clear()
    print(f"Done unloading models.")
    return {"status": "ok"}


@dataclass
class CompletionResponse:
    prompt: str
    response: str


@app.get("/complete/")
def get_completion(*, prompt: str, max_response_length: int = 30) -> CompletionResponse:
    """Returns the completion of the given prompt by a language model with the given capacity."""
    print(
        f"Completion request: {prompt=}, model capacity: {DEFAULT_CAPACITY} parameters."
    )
    response_text: str = get_response_text(
        prompt, max_length=max_response_length, capacity=DEFAULT_CAPACITY
    )
    print(f"Completion response: {response_text}")
    return CompletionResponse(
        prompt=prompt,
        response=response_text,
    )


@dataclass
class EmbeddingResponse:
    prompt: str
    hidden_state: list[float]


@app.get("/embedding/")
def get_embedding(*, prompt: str) -> EmbeddingResponse:
    """Gets the embedding vector for a given prompt."""
    print(
        f"Embedding request: {prompt=}, model capacity: {DEFAULT_CAPACITY} parameters."
    )
    hidden_state = get_hidden_state(prompt, capacity=DEFAULT_CAPACITY)
    return EmbeddingResponse(
        prompt=prompt, hidden_state=hidden_state.flatten().cpu().tolist()
    )
