import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from icp_api.backends import build_backends
from icp_api.backends.base import Backend
from icp_api.config import Settings, get_settings
from icp_api.db import init_db, make_engine
from icp_api.rollout.engine import RolloutEngine, SmokeFn, recover_interrupted
from icp_api.routes import deployments, metrics

log = logging.getLogger("icp")


def create_app(
    settings: Settings | None = None,
    backends: dict[str, Backend] | None = None,
    smoke: SmokeFn | None = None,
) -> FastAPI:
    """Arguments are for tests; in normal use everything comes from ICP_* settings."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cfg = settings or get_settings()
        db = make_engine(cfg.database_url)
        init_db(db)
        if recovered := recover_interrupted(db):
            log.warning("marked %d interrupted rollouts as failed", recovered)
        app.state.settings = cfg
        app.state.db = db
        app.state.backends = backends if backends is not None else build_backends(cfg)
        app.state.rollouts = RolloutEngine(db, app.state.backends, cfg, smoke=smoke)
        yield
        await app.state.rollouts.shutdown()
        db.dispose()

    app = FastAPI(title="Inference Control Plane", version="0.1.0", lifespan=lifespan)
    app.include_router(deployments.router, prefix="/api")
    app.include_router(metrics.router, prefix="/api")

    @app.get("/healthz", include_in_schema=False)
    def healthz() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
