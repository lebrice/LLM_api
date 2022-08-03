""" TODO: Create a script that can be run on the client side:

1. SSH into the cluster if necessary
2. Find the server.txt containing the hostname and port
3. Port forward:  `ssh -nNL 10101:cn-a010:12345 mila`
4. (bonus) Open a browser window at the docs url
"""
import webbrowser
import subprocess
from milatools.

def _get_server_info(remote, identifier, hide=False):
    # TAKEN FROM milatools.cli.__main__
    text = remote.get_output(f"cat .milatools/control/{identifier}", hide=hide)
    info = dict(line.split(" = ") for line in text.split("\n") if line)
    return info
# from milatools.cli.utils import Remote

subprocess.popen(
    "ssh",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "StrictHostKeyChecking=no",
    "-nNL",
    f"localhost:{port}:{to_forward}",
    node,
)
webbrowser.open(url)