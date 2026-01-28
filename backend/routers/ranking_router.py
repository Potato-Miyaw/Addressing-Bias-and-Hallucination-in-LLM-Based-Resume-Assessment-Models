"""
Fairness-Aware Ranking Router
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import sys
import os
from fastapi.responses import StreamingResponse
import io

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature5_xgb_ranker import FairnessAwareRanker
from backend.services.audit_exporter import build_shortlist_csv, build_audit_pdf

router = APIRouter(prefix="/api/rank", tags=["Ranking"])

# Lazy load services
ranker = None

def get_ranker():
    global ranker
    if ranker is None:
        ranker = FairnessAwareRanker()
    return ranker

# Pydantic models
class RankRequest(BaseModel):
    job_id: str
    candidates: List[Dict[str, Any]]
    jd_data: Dict[str, Any]
    use_fairness: bool = True
    fairness_method: Optional[str] = None
    sensitive_attribute: Optional[str] = "gender"
    hire_threshold: Optional[float] = 0.5
    save_to_db: bool = True

class ExportRequest(BaseModel):
    job_id: str
    ranked_candidates: List[Dict[str, Any]]
    fairness_metrics: Dict[str, Any] = {}
    fairness_enabled: bool = True
    hire_threshold: Optional[float] = 0.5

@router.post("/")
async def rank_candidates(request: RankRequest):
    """Rank candidates using XGBoost + Fairlearn"""
    try:
        ranker_service = get_ranker()

        result = ranker_service.rank_candidates_with_metrics(
            request.candidates,
            request.jd_data,
            use_fairness=request.use_fairness,
            fairness_method=request.fairness_method,
            sensitive_attribute=request.sensitive_attribute or "gender",
            hire_threshold=request.hire_threshold if request.hire_threshold is not None else 0.5,
        )

        ranked_candidates = result["ranked_candidates"]
        fairness_metrics = result["fairness_metrics"]

        return {
            "success": True,
            "job_id": request.job_id,
            "total_candidates": len(request.candidates),
            "fairness_enabled": request.use_fairness,
            "fairness_method": request.fairness_method,
            "ranked_candidates": ranked_candidates,
            "fairness_metrics": fairness_metrics,
            "saved_to_db": False
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ranking failed: {str(e)}"
        )

@router.post("/export/shortlist")
async def export_shortlist_csv(request: ExportRequest):
    """Export shortlist CSV with decision and fairness mode."""
    try:
        fairness_mode = request.fairness_metrics.get("fairness_mode") or ("ON" if request.fairness_enabled else "OFF")
        csv_bytes = build_shortlist_csv(
            request.ranked_candidates,
            fairness_mode=fairness_mode,
            hire_threshold=request.hire_threshold if request.hire_threshold is not None else 0.5,
        )

        return StreamingResponse(
            io.BytesIO(csv_bytes),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=shortlist_{request.job_id}.csv"
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV export failed: {str(e)}"
        )

@router.post("/export/audit")
async def export_audit_pdf(request: ExportRequest):
    """Export fairness & hallucination audit PDF."""
    try:
        pdf_bytes = build_audit_pdf(
            request.job_id,
            request.ranked_candidates,
            request.fairness_metrics,
        )
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=audit_{request.job_id}.pdf"
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audit PDF export failed: {str(e)}"
        )
