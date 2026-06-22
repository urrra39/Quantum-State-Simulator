"""FastAPI application entrypoint for TensorQ-Engine."""

from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from tensorq import __version__
from tensorq.api.routes import router as api_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="TensorQ-Engine",
        description=(
            "High-performance quantum circuit simulator. Computes state-vector "
            "evolution under fundamental quantum gates using from-scratch linear "
            "algebra (NumPy/SciPy)."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    origins_env = os.getenv("TENSORQ_CORS_ORIGINS", "http://localhost:3000")
    origins = [o.strip() for o in origins_env.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", tags=["meta"])
    def root() -> dict[str, str]:
        return {
            "service": "TensorQ-Engine",
            "version": __version__,
            "docs": "/docs",
        }

    return app


app = create_app()
