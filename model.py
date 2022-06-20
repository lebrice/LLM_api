import functools
import os
from pathlib import Path
from typing import Literal
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.opt.modeling_opt import OPTModel, OPTForCausalLM
from torch import Tensor
import torch

tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-350m")
Capacity = Literal["350m", "1.3b", "2.7b", "13b", "30b"]


def _get_default_capacity() -> Capacity:
    if "DEFAULT_CAPACITY" in os.environ:
        capacity_env_variable = os.environ["DEFAULT_CAPACITY"]
        if capacity_env_variable not in ["350m", "1.3b", "2.7b", "13b", "30b"]:
            raise ValueError(
                f"CAPACITY={capacity_env_variable} is not a valid capacity"
            )
        return capacity_env_variable  # type: ignore
    return "2.7b"


DEFAULT_CAPACITY: Capacity = _get_default_capacity()
OFFLOAD_FOLDER: Path = Path(os.environ.get("SLURM_TMPDIR", "offload"))


@functools.cache
def load_embedding_model(capacity: Capacity = DEFAULT_CAPACITY) -> OPTModel:
    pretrained_embedding_model = OPTModel.from_pretrained(
        f"facebook/opt-{capacity}",
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=OFFLOAD_FOLDER,
    )
    assert isinstance(pretrained_embedding_model, OPTModel)
    return pretrained_embedding_model


@functools.cache
def load_completion_model(capacity: Capacity = DEFAULT_CAPACITY) -> OPTForCausalLM:
    pretrained_causal_lm_model = OPTForCausalLM.from_pretrained(
        f"facebook/opt-{capacity}",
        device_map="auto",
        torch_dtype=torch.float16,
        offload_folder=OFFLOAD_FOLDER,
    )
    assert isinstance(pretrained_causal_lm_model, OPTForCausalLM)
    return pretrained_causal_lm_model


@functools.cache
def load_tokenizer(capacity: Capacity = DEFAULT_CAPACITY) -> GPT2Tokenizer:
    pretrained_tokenizer = GPT2Tokenizer.from_pretrained(
        f"facebook/opt-{capacity}",
        device_map="auto",
        torch_dtype=torch.float16,
    )
    assert isinstance(pretrained_tokenizer, GPT2Tokenizer)
    return pretrained_tokenizer


def tokenize(prompt: str, capacity: Capacity = DEFAULT_CAPACITY) -> BatchEncoding:
    tokenizer = load_tokenizer(capacity=capacity)
    return tokenizer(prompt, return_tensors="pt")


@torch.no_grad()
def get_hidden_state(prompt: str, capacity: Capacity = DEFAULT_CAPACITY) -> Tensor:
    inputs = tokenize(prompt)
    model = load_embedding_model()
    outputs = model(**inputs.to(model.device))

    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states


@torch.no_grad()
def get_response_text(
    prompt: str, max_length: int = 30, capacity: Capacity = DEFAULT_CAPACITY
) -> str:
    inputs = tokenize(prompt)
    model = load_completion_model(capacity=capacity)
    generate_ids = model.generate(
        inputs.input_ids.to(model.device), max_length=max_length
    )
    prompt_and_response = tokenizer.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    assert isinstance(prompt_and_response, str)
    model_response = prompt_and_response.replace(prompt, "").lstrip()
    return model_response
