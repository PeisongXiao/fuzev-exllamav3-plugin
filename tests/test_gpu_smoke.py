"""Opt-in SystemOne HTTP smoke against an existing GPU server."""

import os

import httpx
import pytest


@pytest.mark.gpu
def test_systemone_on_running_native_service():
    base_url = os.environ.get("FUZEV_NATIVE_SMOKE_URL")
    if not base_url:
        pytest.skip("Set FUZEV_NATIVE_SMOKE_URL to an already running GPU-backed service")
    headers = {}
    if os.environ.get("FUZEV_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["FUZEV_API_KEY"]
    with httpx.Client(base_url=base_url, headers=headers, timeout=120) as client:
        models = client.get("/v1/models")
        models.raise_for_status()
        response = client.post("/v1/systemone", json={
            "model": models.json()["data"][0]["id"], "state": "The invoice was paid twice.",
            "questions": {"duplicate": {"type": "noul", "instructions": "Was payment duplicated?"}}})
    assert response.status_code == 200, response.text
    assert 0 <= response.json()["answers"]["duplicate"]["noul"] <= 1
