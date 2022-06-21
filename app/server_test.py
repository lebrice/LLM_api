from fastapi import FastAPI
from fastapi.testclient import TestClient
from server import app


client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_completion():
    response = client.get(
        "/complete/",
        # headers={"X-Token": "coneofsilence"},
        json={"prompt": "Hey there! How's it going?"},
    )
    assert response.status_code == 200, response
    assert response.json() == {
        "id": "foobar",
        "title": "Foo Bar",
        "description": "The Foo Barters",
    }
