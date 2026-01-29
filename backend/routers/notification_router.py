"""Notification router for Teams webhook."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from backend.services.teams_notifier import build_summary, post_to_teams
from backend.services.audit_exporter import build_shortlist_csv, build_audit_pdf
from backend.services.power_automate_notifier import send_to_power_automate

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


@router.post("/power-automate")
async def notify_power_automate(request: TeamsNotifyRequest):
    try:
        summary = build_summary(
            job_id=request.job_id,
            ranked_candidates=request.ranked_candidates,
            fairness_metrics=request.fairness_metrics,
            match_threshold=request.match_threshold if request.match_threshold is not None else 0.7,
            hire_threshold=request.hire_threshold if request.hire_threshold is not None else 0.5,
        )

        fairness_mode = summary.get("fairness_metrics", {}).get("fairness_mode", "OFF")
        shortlist_csv = build_shortlist_csv(
            request.ranked_candidates,
            fairness_mode=fairness_mode,
            hire_threshold=request.hire_threshold if request.hire_threshold is not None else 0.5,
        )
        audit_pdf = build_audit_pdf(
            request.job_id,
            request.ranked_candidates,
            request.fairness_metrics or {},
        )

        send_to_power_automate(summary, shortlist_csv, audit_pdf)
        return {"success": True, "summary": summary}
    except RuntimeError as exc:
        # Configuration errors should return 400 Bad Request with helpful message
        if "not configured" in str(exc):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Power Automate notification failed: {str(exc)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Power Automate notification failed: {str(exc)}",
        )
