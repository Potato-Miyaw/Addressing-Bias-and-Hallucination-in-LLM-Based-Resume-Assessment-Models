"""Notification router for Teams webhook."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from backend.services.teams_notifier import build_summary, post_to_teams

router = APIRouter(prefix="/api/notify", tags=["Notifications"])


class TeamsNotifyRequest(BaseModel):
    job_id: str
    ranked_candidates: List[Dict[str, Any]]
    fairness_metrics: Dict[str, Any] = {}
    match_threshold: Optional[float] = 0.7
    hire_threshold: Optional[float] = 0.5


@router.post("/teams")
async def notify_teams(request: TeamsNotifyRequest):
    try:
        summary = build_summary(
            job_id=request.job_id,
            ranked_candidates=request.ranked_candidates,
            fairness_metrics=request.fairness_metrics,
            match_threshold=request.match_threshold if request.match_threshold is not None else 0.7,
            hire_threshold=request.hire_threshold if request.hire_threshold is not None else 0.5,
        )
        post_to_teams(summary)
        return {"success": True, "summary": summary}
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Teams notification failed: {str(exc)}",
        )
