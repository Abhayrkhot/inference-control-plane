"""Deployments, revisions, rollouts, rollbacks, events and logs."""

from collections.abc import AsyncIterator, Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import StreamingResponse
from sqlmodel import Session, col, select

from icp_api.models import Deployment, Revision, RevisionStatus, RolloutEvent
from icp_api.rollout.engine import revision_ref
from icp_api.schemas import (
    DeploymentCreate,
    DeploymentOut,
    DeploymentSpec,
    EventOut,
    RevisionOut,
    RollbackRequest,
)

router = APIRouter(tags=["deployments"])
IN_FLIGHT = (RevisionStatus.PENDING, RevisionStatus.ROLLING_OUT)


def get_session(request: Request) -> Iterator[Session]:
    with Session(request.app.state.db) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session)]


def find_deployment(s: Session, name: str) -> Deployment:
    deployment = s.exec(select(Deployment).where(Deployment.name == name)).first()
    if deployment is None:
        raise HTTPException(404, f"deployment {name!r} not found")
    return deployment


def find_revision(s: Session, deployment: Deployment, number: int) -> Revision:
    rev = s.exec(select(Revision).where(Revision.deployment_id == deployment.id, Revision.number == number)).first()
    if rev is None:
        raise HTTPException(404, f"{deployment.name} has no revision r{number}")
    return rev


def revisions_of(s: Session, deployment: Deployment) -> list[Revision]:
    query = select(Revision).where(Revision.deployment_id == deployment.id).order_by(col(Revision.number).desc())
    return list(s.exec(query).all())


def revision_out(rev: Revision) -> RevisionOut:
    return RevisionOut.model_validate(rev, from_attributes=True)


def deployment_out(s: Session, deployment: Deployment) -> DeploymentOut:
    active = s.get(Revision, deployment.active_revision_id) if deployment.active_revision_id else None
    revisions = revisions_of(s, deployment)
    return DeploymentOut(
        name=deployment.name,
        created_at=deployment.created_at,
        active_revision=revision_out(active) if active else None,
        latest_revision=revision_out(revisions[0]) if revisions else None,
    )


def check_backend(request: Request, spec: DeploymentSpec) -> None:
    if spec.backend not in request.app.state.backends:
        raise HTTPException(400, f"backend {spec.backend!r} is not configured")


def start_rollout(
    request: Request, s: Session, deployment: Deployment, spec: DeploymentSpec, source_revision: int | None = None
) -> Revision:
    check_backend(request, spec)
    revisions = revisions_of(s, deployment)
    busy = next((r for r in revisions if r.status in IN_FLIGHT), None)
    if busy:
        raise HTTPException(409, f"r{busy.number} is still rolling out")
    rev = Revision(
        deployment_id=deployment.id,
        number=revisions[0].number + 1 if revisions else 1,
        backend=spec.backend,
        spec=spec.model_dump(mode="json"),
        source_revision=source_revision,
    )
    s.add(rev)
    s.commit()
    s.refresh(rev)
    request.app.state.rollouts.start(rev.id)
    return rev


@router.get("/deployments", response_model=list[DeploymentOut])
def list_deployments(s: SessionDep) -> list[DeploymentOut]:
    return [deployment_out(s, d) for d in s.exec(select(Deployment).order_by(Deployment.name)).all()]


@router.post("/deployments", response_model=DeploymentOut, status_code=201)
async def create_deployment(body: DeploymentCreate, request: Request, s: SessionDep) -> DeploymentOut:
    check_backend(request, body.spec)
    if s.exec(select(Deployment).where(Deployment.name == body.name)).first():
        raise HTTPException(409, f"deployment {body.name!r} already exists; roll out a new revision instead")
    deployment = Deployment(name=body.name)
    s.add(deployment)
    s.commit()
    s.refresh(deployment)
    start_rollout(request, s, deployment, body.spec)
    return deployment_out(s, deployment)


@router.get("/deployments/{name}", response_model=DeploymentOut)
def get_deployment(name: str, s: SessionDep) -> DeploymentOut:
    return deployment_out(s, find_deployment(s, name))


@router.delete("/deployments/{name}", status_code=204)
async def delete_deployment(name: str, request: Request, s: SessionDep) -> Response:
    deployment = find_deployment(s, name)
    revisions = revisions_of(s, deployment)
    if any(r.status in IN_FLIGHT for r in revisions):
        raise HTTPException(409, "a rollout is in progress; wait for it to finish")
    for rev in revisions:
        backend = request.app.state.backends.get(rev.backend)
        if rev.status == RevisionStatus.ACTIVE and backend is not None:
            await backend.deactivate(deployment.name)
            await backend.teardown(revision_ref(rev, deployment.name))
    for event in s.exec(select(RolloutEvent).where(RolloutEvent.deployment_id == deployment.id)).all():
        s.delete(event)
    for rev in revisions:
        s.delete(rev)
    s.delete(deployment)
    s.commit()
    return Response(status_code=204)


@router.post("/deployments/{name}/stop", response_model=DeploymentOut)
async def stop_deployment(name: str, request: Request, s: SessionDep) -> DeploymentOut:
    """Tear down the serving revision and free its resources, but keep the deployment's
    revisions, events and timings. Roll out or roll back later to serve again."""
    deployment = find_deployment(s, name)
    if any(r.status in IN_FLIGHT for r in revisions_of(s, deployment)):
        raise HTTPException(409, "a rollout is in progress; wait for it to finish")
    active = s.get(Revision, deployment.active_revision_id) if deployment.active_revision_id else None
    if active is None:
        raise HTTPException(400, f"{name} is not serving anything")
    backend = request.app.state.backends.get(active.backend)
    if backend is None:
        raise HTTPException(400, f"backend {active.backend!r} is not configured")
    await backend.deactivate(deployment.name)
    await backend.teardown(revision_ref(active, deployment.name))
    active.status = RevisionStatus.STOPPED
    deployment.active_revision_id = None
    s.add(active)
    s.add(deployment)
    s.add(
        RolloutEvent(
            revision_id=active.id,
            deployment_id=deployment.id,
            kind="stopped",
            message=f"r{active.number} stopped by the operator; its resources were released",
        )
    )
    s.commit()
    return deployment_out(s, deployment)


@router.post("/deployments/{name}/rollouts", response_model=RevisionOut, status_code=202)
async def create_rollout(name: str, spec: DeploymentSpec, request: Request, s: SessionDep) -> RevisionOut:
    return revision_out(start_rollout(request, s, find_deployment(s, name), spec))


@router.post("/deployments/{name}/rollback", response_model=RevisionOut, status_code=202)
async def rollback(name: str, request: Request, s: SessionDep, body: RollbackRequest | None = None) -> RevisionOut:
    """Roll out an earlier revision's spec again, as a new revision."""
    deployment = find_deployment(s, name)
    if body and body.revision is not None:
        target = find_revision(s, deployment, body.revision)
        if target.status == RevisionStatus.FAILED:
            raise HTTPException(400, f"r{target.number} never served traffic; pick one that did")
        if target.id == deployment.active_revision_id:
            raise HTTPException(400, f"r{target.number} is already serving")
    else:
        served = (RevisionStatus.SUPERSEDED, RevisionStatus.STOPPED)
        target = next((r for r in revisions_of(s, deployment) if r.status in served), None)
        if target is None:
            raise HTTPException(400, "no earlier revision that served traffic")
    spec = DeploymentSpec.model_validate(target.spec)
    return revision_out(start_rollout(request, s, deployment, spec, source_revision=target.number))


@router.get("/deployments/{name}/revisions", response_model=list[RevisionOut])
def list_revisions(name: str, s: SessionDep) -> list[RevisionOut]:
    return [revision_out(r) for r in revisions_of(s, find_deployment(s, name))]


@router.get("/deployments/{name}/events", response_model=list[EventOut])
def list_events(
    name: str,
    s: SessionDep,
    after: int = 0,
    revision: int | None = None,
    limit: Annotated[int, Query(ge=1, le=5000)] = 1000,
) -> list[EventOut]:
    """Events oldest first. Poll with `after` set to the last id you saw."""
    deployment = find_deployment(s, name)
    query = (
        select(RolloutEvent, Revision.number)
        .join(Revision, col(RolloutEvent.revision_id) == col(Revision.id))
        .where(RolloutEvent.deployment_id == deployment.id, col(RolloutEvent.id) > after)
    )
    if revision is not None:
        query = query.where(Revision.number == revision)
    rows = s.exec(query.order_by(col(RolloutEvent.id)).limit(limit)).all()
    return [EventOut(id=e.id, revision=n, at=e.at, kind=e.kind, message=e.message) for e, n in rows]


@router.get("/deployments/{name}/logs")
async def stream_logs(
    name: str,
    request: Request,
    s: SessionDep,
    revision: int | None = None,
    follow: bool = False,
    tail: Annotated[int, Query(ge=1, le=5000)] = 200,
) -> StreamingResponse:
    """Logs of the given revision; by default the active one, else the latest."""
    deployment = find_deployment(s, name)
    if revision is not None:
        rev = find_revision(s, deployment, revision)
    else:
        active = s.get(Revision, deployment.active_revision_id) if deployment.active_revision_id else None
        rev = active or next(iter(revisions_of(s, deployment)), None)
    if rev is None:
        raise HTTPException(404, f"{name} has no revisions")
    backend = request.app.state.backends.get(rev.backend)
    if backend is None:
        raise HTTPException(400, f"backend {rev.backend!r} is not configured")
    ref = revision_ref(rev, deployment.name)

    async def body() -> AsyncIterator[str]:
        try:
            async for chunk in backend.logs(ref, follow=follow, tail=tail):
                yield chunk
        except Exception as e:
            yield f"\n[icp] log stream ended: {type(e).__name__}: {e}\n"

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8")
