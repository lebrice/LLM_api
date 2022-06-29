# FastAPI + HuggingFace + SLURM

Proof-of-concept for an API for performing inference with a Large Language Model (LLM).

## Installation:

```console
> conda create -n llm python=3.10
> conda activate llm
> pip install git+https://www.github.com/lebrice/LLM_api.git
```

## Usage:

Available options:
```console
> python app/server.py --help
usage: server.py [-h] [--model_capacity str] [--hf_cache_dir Path] [--port int]
                 [--reload bool] [--offload_folder Path]

 API for querying a large language model. 

options:
  -h, --help            show this help message and exit

Settings ['settings']:
   Configuration settings for the API. 

  --model_capacity str  Model capacity. (default: 13b)
  --hf_cache_dir Path   (default: ~/scratch/cache/huggingface)
  --port int            The port to run the server on. (default: 12345)
  --reload bool         Wether to restart the server (and reload the model) when the
                        source code changes. (default: False)
  --offload_folder Path
                        Folder where the model weights will be offloaded if the entire
                        model doesn't fit in memory. (default: /Tmp/slurm.1968686.0)
```

Spinning up the server:
```console
> python app/server.py
HF_HOME='/home/mila/n/normandf/scratch/cache/huggingface'
TRANSFORMERS_CACHE='/home/mila/n/normandf/scratch/cache/huggingface/transformers'
Running the server with the following settings: {"model_capacity": "13b", "hf_cache_dir": "~/scratch/cache/huggingface", "port": 12345, "reload": false, "offload_folder": "/Tmp/slurm.1968686.0"}
INFO:     Started server process [25042]
INFO:     Waiting for application startup.
Writing address_string='cn-b003:8000' to server.txt
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:12345 (Press CTRL+C to quit)
```

(WIP) Run as a slurm job:

```console
> sbatch run_server.sh
```

(WIP) Connecting to a running job:

```python
import time
from app.client import server_is_up, get_completion_text
while not server_is_up():
    print("Waiting for the server to be online...")
    time.sleep(10)
print("server is up!")
rest_of_story = get_completion_text("Once upon a time, there lived a great wizard.")
```