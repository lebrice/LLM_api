import functools
import os
from pathlib import Path
from typing import Literal
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.opt.modeling_opt import OPTForCausalLM
import torch
import logging

from fastapi import Depends, FastAPI

from dataclasses import dataclass
import socket
import os
from fastapi import Request
from logging import getLogger as get_logger
from fastapi.responses import RedirectResponse

from simple_parsing import field
from pydantic import BaseSettings

Capacity = Literal["350m", "1.3b", "2.7b", "13b", "30b"]


@dataclass(init=False)
class ServerConfig(BaseSettings):
    port: int = 12345

    reload: bool = False
    model_capacity: str = "13b"
    # choices: ["350m", "1.3b", "2.7b", "13b", "30b"]

    offload_folder: Path = Path(os.environ.get("SLURM_TMPDIR", "model_offload"))


""" API for querying a large language model. """
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


app = FastAPI(
    on_startup=[
        write_server_address_to_file,
    ]
)


@app.get("/")
def root(request: Request):
    return RedirectResponse(url=f"{request.base_url}docs")


@dataclass
class CompletionResponse:
    prompt: str
    response: str


@functools.cache
def get_settings() -> ServerConfig:
    return ServerConfig()


@app.get("/complete/")
async def get_completion(
    prompt: str,
    max_response_length: int = 30,
    settings: ServerConfig = Depends(get_settings),
) -> CompletionResponse:
    """Returns the completion of the given prompt by a language model with the given capacity."""
    capacity = settings.model_capacity
    offload_folder = settings.offload_folder
    print(f"Completion request: {prompt=}, model capacity: {capacity} parameters.")

    model = load_completion_model(capacity=capacity, offload_folder=offload_folder)
    tokenizer = load_tokenizer(capacity=capacity)

    response_text = get_response_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_response_length=max_response_length,
    )

    print(f"Completion response: {response_text}")
    return CompletionResponse(
        prompt=prompt,
        response=response_text,
    )


@functools.cache
def load_completion_model(capacity: Capacity, offload_folder: Path) -> OPTForCausalLM:
    print(f"Loading OPT completion model with {capacity} parameters...")
    pretrained_causal_lm_model = OPTForCausalLM.from_pretrained(
        f"facebook/opt-{capacity}",
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=offload_folder,
    )
    assert isinstance(pretrained_causal_lm_model, OPTForCausalLM)
    print("Done.")
    return pretrained_causal_lm_model


@functools.cache
def load_tokenizer(capacity: Capacity) -> GPT2Tokenizer:
    print(f"Loading Tokenizer for model with {capacity} parameters...")
    pretrained_tokenizer = GPT2Tokenizer.from_pretrained(
        f"facebook/opt-{capacity}",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    assert isinstance(pretrained_tokenizer, GPT2Tokenizer)
    return pretrained_tokenizer


@torch.no_grad()
def get_response_text(
    model: OPTForCausalLM,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    max_response_length: int = 30,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    print(f"Generating based on {prompt=}...")
    generate_ids = model.generate(
        inputs.input_ids.to(model.device), max_length=max_response_length
    )
    prompt_and_response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    assert isinstance(prompt_and_response, str)
    model_response = prompt_and_response.replace(prompt, "").lstrip()
    return model_response


# @torch.no_grad()
# def get_hidden_state(prompt: str, capacity: Capacity = DEFAULT_CAPACITY) -> Tensor:
#     inputs = tokenize(prompt)
#     model = load_embedding_model()
#     outputs = model(**inputs.to(model.device))

#     last_hidden_states = outputs.last_hidden_state
#     return last_hidden_states
