"""
Claim Verification Router with Automatic Ground Truth Extraction
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
from backend.services.feature3_ground_truth_extractor import GroundTruthExtractor

router = APIRouter(prefix="/api/verify", tags=["Verification"])

# Lazy load services
claim_verifier = None
ground_truth_extractor = None

def get_claim_verifier():
    global claim_verifier
    if claim_verifier is None:
        claim_verifier = ClaimVerifier()
    return claim_verifier

def get_ground_truth_extractor():
    global ground_truth_extractor
    if ground_truth_extractor is None:
        ground_truth_extractor = GroundTruthExtractor()
    return ground_truth_extractor

# Pydantic models
class VerifyClaimRequest(BaseModel):
    extraction: Any
    ground_truth: Optional[Any] = None

class VerifyResumeRequest(BaseModel):
    resume_id: str
    resume_extractions: Dict[str, Any]
    ground_truth_data: Optional[Dict[str, Any]] = None
    auto_extract_ground_truth: bool = True  # NEW: Automatically extract from resume text

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
    """
    Verify all claims in a resume extraction
    
    NEW: Automatically extracts ground truth from raw resume text if available
    This checks if LLM extractions actually appear in the original resume
    """
    try:
        verifier = get_claim_verifier()
        ground_truth_data = request.ground_truth_data
        
        # AUTO-EXTRACT GROUND TRUTH FROM RAW TEXT
        if request.auto_extract_ground_truth and 'text' in request.resume_extractions:
            print("🔍 Auto-extracting ground truth from resume text...")
            
            gt_extractor = get_ground_truth_extractor()
            extracted_gt = gt_extractor.extract_ground_truth_from_text(
                request.resume_extractions['text'],
                request.resume_extractions
            )
            
            # Merge with provided ground truth (provided takes precedence)
            if ground_truth_data:
                ground_truth_data.update(extracted_gt)
            else:
                ground_truth_data = extracted_gt
            
            print(f"✅ Extracted {len(extracted_gt)} ground truth fields from text")
        
        # Verify claims
        report = verifier.verify_resume_data(
            request.resume_extractions,
            ground_truth_data
        )
        
        report['resume_id'] = request.resume_id
        report['auto_extracted_ground_truth'] = request.auto_extract_ground_truth and 'text' in request.resume_extractions
        
        return {
            "success": True,
            "resume_id": request.resume_id,
            "verification_report": report
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume verification failed: {str(e)}"
        )

@router.post("/extract-ground-truth")
async def extract_ground_truth(
    resume_text: str,
    resume_extractions: Dict[str, Any]
):
    """
    Extract ground truth from raw resume text
    Useful for debugging and seeing what ground truth would be extracted
    """
    try:
        gt_extractor = get_ground_truth_extractor()
        ground_truth = gt_extractor.extract_ground_truth_from_text(
            resume_text,
            resume_extractions
        )
        
        return {
            "success": True,
            "ground_truth": ground_truth,
            "fields_extracted": len(ground_truth)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ground truth extraction failed: {str(e)}"
        )