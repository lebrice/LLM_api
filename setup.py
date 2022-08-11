from setuptools import find_namespace_packages, setup

setup(
    name="mila_transformers",
    version="0.0.1",
    author="Fabrice Normandin",
    author_email="normandf@mila.quebec",
    packages=find_namespace_packages("app"),
    install_requires=[
        "fastapi",
        "transformers",
        "uvicorn[standard]",
        "accelerate",
        "simple-parsing",
    ],
)
