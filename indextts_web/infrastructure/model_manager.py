"""Provisioning API, separated from GPU inference startup."""

from __future__ import annotations

import threading
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from indextts_web.infrastructure.model_setup import validate_operation


class Operation(BaseModel):
    action: str
    target: str


def create_manager_app(*, get_state, execute_operation) -> FastAPI:
    app = FastAPI(title="Audio Studio · Models & Setup")
    state_lock = threading.Lock()
    jobs = {}
    cached_state = {"repositories": [], "models": [], "environments": []}

    def run_operation(job):
        nonlocal cached_state

        def emit(line):
            print(line, flush=True)
            job["logs"] = [*job["logs"], line][-200:]

        # Preparation and status reloads share a lock because Volume.reload()
        # cannot run while a download has open files. Polling stays responsive.
        with state_lock:
            job["status"] = "running"
            final_status = "completed"
            try:
                execute_operation(job["action"], job["target"], emit)
            except Exception as exc:
                final_status = "failed"
                job["error"] = str(exc)
                emit(str(exc))
            try:
                cached_state = get_state()
            except Exception as exc:
                emit(f"Refresh failed: {exc}")
            job["status"] = final_status

    @app.middleware("http")
    async def prevent_caching(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (Path(__file__).parents[2] / "static/model_manager.html").read_text(encoding="utf-8")

    @app.get("/api/state")
    def state():
        nonlocal cached_state
        if state_lock.acquire(blocking=False):
            try:
                cached_state = get_state()
            finally:
                state_lock.release()
        return cached_state

    @app.post("/api/jobs", status_code=202)
    def start(operation: Operation, background_tasks: BackgroundTasks):
        try:
            validate_operation(operation.action, operation.target)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        job = {"id": uuid.uuid4().hex, "action": operation.action, "target": operation.target,
               "status": "queued", "logs": []}
        jobs[job["id"]] = job
        background_tasks.add_task(run_operation, job)
        return dict(job)

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        result = jobs.get(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found or expired")
        return result

    return app
