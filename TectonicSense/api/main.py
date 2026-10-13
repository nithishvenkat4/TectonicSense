"""TectonicSense FastAPI service.

Inference endpoints will be implemented after the final model is trained.
"""

from fastapi import FastAPI

app = FastAPI(
    title="TectonicSense API",
    description="Tectonic environment classification using explainable ML.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "project": "TectonicSense"}
