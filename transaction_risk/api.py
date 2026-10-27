"""Score API. Loads the trained baseline-v1 artifact at startup."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from transaction_risk import __version__
from transaction_risk.schema import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    ScoreResponse,
    Transaction,
)
from transaction_risk.scoring import load_artifact, score_payloads


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.artifact = load_artifact()
        yield

    app = FastAPI(
        title="Transaction risk",
        version=__version__,
        description=(
            "Score synthetic card-not-present transactions with the baseline-v1 "
            "logistic regression. The request schema is closed: only the documented "
            "transaction fields are accepted."
        ),
        lifespan=lifespan,
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> dict:
        artifact = app.state.artifact
        return {"status": "ok", "model_version": artifact["model_version"]}

    @app.post("/v1/score", response_model=ScoreResponse)
    def score_one(body: Transaction) -> dict:
        return score_payloads(app.state.artifact, [body.model_dump()])[0]

    @app.post("/v1/score/batch", response_model=BatchResponse)
    def score_batch(body: BatchRequest) -> dict:
        payloads = [item.model_dump() for item in body.transactions]
        return {"results": score_payloads(app.state.artifact, payloads)}

    return app


app = create_app()
