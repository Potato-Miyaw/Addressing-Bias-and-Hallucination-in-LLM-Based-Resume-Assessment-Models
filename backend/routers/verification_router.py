"""
Claim Verification Router
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature3_claim_verifier import ClaimVerifier

router = APIRouter(prefix="/api/verify", tags=["Verification"])

# Lazy load verifier
claim_verifier = None

def get_claim_verifier():
    global claim_verifier
    if claim_verifier is None:
        claim_verifier = ClaimVerifier()
    return claim_verifier

# Pydantic models
class VerifyClaimRequest(BaseModel):
    extraction: Any
    ground_truth: Optional[Any] = None

class VerifyResumeRequest(BaseModel):
    resume_id: str
    resume_extractions: Dict[str, Any]
    ground_truth_data: Optional[Dict[str, Any]] = None

@router.post("/claim")
async def verify_single_claim(request: VerifyClaimRequest):
    """Verify a single claim for hallucinations"""
    try:
        verifier = get_claim_verifier()
        result = verifier.verify_claim(
            request.extraction,
            request.ground_truth or ""
        )
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Verification failed: {str(e)}"
        )

@router.post("/resume")
async def verify_resume(request: VerifyResumeRequest):
    """Verify all claims in a resume extraction"""
    try:
        verifier = get_claim_verifier()
        report = verifier.verify_resume_data(
            request.resume_extractions,
            request.ground_truth_data
        )
        
        report['resume_id'] = request.resume_id
        
        return {
            "success": True,
            "resume_id": request.resume_id,
            "verification_report": report
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume verification failed: {str(e)}"
        )