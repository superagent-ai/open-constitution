from __future__ import annotations

from fastapi import FastAPI

from .routes import router

app = FastAPI(
    title="Open Constitution Training API",
    version="1.0.0",
    description="Launch probe and classifier training on Modal and publish artifacts.",
)
app.include_router(router)


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}
