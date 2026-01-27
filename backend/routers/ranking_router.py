"""
Fairness-Aware Ranking Router
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature5_xgb_ranker import FairnessAwareRanker

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
    save_to_db: bool = True

@router.post("/")
async def rank_candidates(request: RankRequest):
    """Rank candidates using XGBoost + Fairlearn"""
    try:
        ranker_service = get_ranker()
        
        ranked_candidates = ranker_service.rank_candidates(
            request.candidates,
            request.jd_data,
            use_fairness=request.use_fairness
        )
        
        return {
            "success": True,
            "job_id": request.job_id,
            "total_candidates": len(request.candidates),
            "fairness_enabled": request.use_fairness,
            "ranked_candidates": ranked_candidates,
            "saved_to_db": False
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ranking failed: {str(e)}"
        )