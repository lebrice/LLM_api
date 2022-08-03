
""" API for querying a large language model. """
from __future__ import annotations

import functools
import logging
import os
import socket
from dataclasses import asdict, dataclass
from logging import getLogger as get_logger
from pathlib import Path
from typing import Literal
import torch
import uvicorn
from fastapi import Depends, FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseSettings
from simple_parsing import ArgumentParser, choice
from simple_parsing.helpers import field
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.opt.modeling_opt import OPTForCausalLM

Capacity = Literal["350m", "1.3b", "2.7b", "13b", "30b"]

# TODO: Setup logging correctly with FastAPI.
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


def get_slurm_tmpdir() -> Path | None:
    """ Returns the slurm temporary directory, if known.
    
    This also works with `mila code`, since the vscode terminals that you get with that command
    don't have all the SLURM env variables, they only have SLURM_JOB_ID.
    """
    if "SLURM_TMPDIR" in os.environ:
        return Path(os.environ["SLURM_TMPDIR"])
    if "SLURM_JOB_ID" in os.environ:
        job_id = os.environ["SLURM_JOB_ID"]
        slurm_tmpdir = Path(f"/Tmp/slurm.{job_id}.0")
        if slurm_tmpdir.is_dir():
            return slurm_tmpdir
    return None


@dataclass(init=False)
class Settings(BaseSettings):
    """ Configuration settings for the API. """

    model_capacity: str = choice("350m", "1.3b", "2.7b", "13b", "30b", default="13b")
    """ Model capacity. """

    hf_cache_dir: Path = Path("~/scratch/cache/huggingface")
    
    port: int = 12345
    """ The port to run the server on."""

    reload: bool = False
    """ Wether to restart the server (and reload the model) when the source code changes. """
    
    model_capacity: str = "13b"
    """ Version of the OPT model to use. Can be one of "350m", "1.3b", "2.7b", "13b", or "30b". """

    offload_folder: Path = Path(get_slurm_tmpdir() or "model_offload")
    """
    Folder where the model weights will be offloaded if the entire model doesn't fit in memory.
    """




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
    ],
    title="SLURM + FastAPI + HuggingFace",
    dependencies=[]
)

@app.get("/")
def root(request: Request):
    return RedirectResponse(url=f"{request.base_url}docs")


@dataclass
class CompletionResponse:
    prompt: str
    response: str


@functools.cache
def get_settings() -> Settings:
    return Settings()


def preload_components(settings: Settings = Depends(get_settings)):
    print(f"Preloading components: {settings=}")
    load_completion_model(capacity=settings.model_capacity, offload_folder=settings.offload_folder)
    load_tokenizer(capacity=settings.model_capacity)


@app.get("/complete/")
async def get_completion(
    prompt: str,
    max_response_length: int = 30,
    settings: Settings = Depends(get_settings),
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

# TODO: Check with students what kind of functionality they want, e.g. extracting representations:
# @torch.no_grad()
# def get_hidden_state(prompt: str, capacity: Capacity = DEFAULT_CAPACITY) -> Tensor:
#     inputs = tokenize(prompt)
#     model = load_embedding_model()
#     outputs = model(**inputs.to(model.device))

#     last_hidden_states = outputs.last_hidden_state
#     return last_hidden_states

# TODO: Look into this `APISettings` class.
# from fastapi_utils.api_settings import get_api_settings, APISettings

# def get_app() -> FastAPI:
#     get_api_settings.cache_clear()
#     settings = get_api_settings()
#     app = FastAPI(**settings.fastapi_kwargs)
#     # <Typically, you would include endpoint routers here>
#     return app

# TODO: Add a training example!

def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_arguments(Settings, "settings", default=Settings())
    args = parser.parse_args()
    settings: Settings = args.settings

    HF_HOME = os.environ.setdefault("HF_HOME", str(settings.hf_cache_dir))
    TRANSFORMERS_CACHE = os.environ.setdefault("TRANSFORMERS_CACHE", str(settings.hf_cache_dir / "transformers"))
    print(f"{HF_HOME=}")
    print(f"{TRANSFORMERS_CACHE=}")

    print(f"Running the server with the following settings: {settings.json()}")

    # NOTE: Can't use `reload` or `workers` when passing the app by value.
    if not settings.reload:
        app.dependency_overrides[get_settings] = lambda: settings
    else:
        # NOTE: If we we want to use `reload=True`, we set the environment variables, so they are
        # used when that module gets imported.
        for k, v in asdict(settings).items():
            os.environ[k.upper()] = str(v) 
    # ssh -nNL 10101:cn-a010:12345 mila
    uvicorn.run(
        (app if not settings.reload else "app.server:app"),  # type: ignore
        port=settings.port,
        # host=socket.gethostname(),
        log_level="debug",
        reload=settings.reload,
    )


if __name__ == "__main__":
    main()
