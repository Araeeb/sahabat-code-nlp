"""HTTP API for Sahabat Code.

    uvicorn api:app --port 8000      (or: make serve)
    POST /v1/analyze  {"text": "..."}
    GET  /health
"""
from fastapi import FastAPI
from pydantic import BaseModel

from sahabat.pipeline import SahabatPipeline
from sahabat.preprocessing import PREPROC_VERSION

app = FastAPI(title="Sahabat Code API", version="1.0.0")
pipeline = SahabatPipeline()          # canary is checked here, at startup


class Request(BaseModel):
    text: str
    k: int = 3


@app.get("/health")
def health():
    return {"status": "ok", "preproc_version": PREPROC_VERSION,
            "model": "camelbert" if pipeline.classifier is not None else "baseline"}


@app.post("/v1/analyze")
def analyze(req: Request):
    return pipeline.analyze(req.text, k=req.k)
