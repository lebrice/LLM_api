from setuptools import setup

setup(
    name="mila_transformers",
    version="0.0.1",
    install_requires=[
        "fastapi",
        "transformers",
        "uvicorn[standard]",
        "accelerate",
    ],
)
