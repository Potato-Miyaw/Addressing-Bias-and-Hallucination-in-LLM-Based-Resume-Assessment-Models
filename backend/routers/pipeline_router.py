"""
Complete Pipeline Router
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature1_jd_extractor import JDExtractor
from backend.services.feature2_bert_ner import ResumeNERExtractor
from backend.services.feature3_claim_verifier import ClaimVerifier
from backend.services.feature4_matcher import JobResumeMatcher
from backend.services.feature5_xgb_ranker import FairnessAwareRanker

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])

# Pydantic model for request body
class CompletePipelineRequest(BaseModel):
    jd_text: str
    resume_texts: List[str]
    use_fairness: bool = False

@router.post("/complete")
async def complete_pipeline(request: CompletePipelineRequest):
    """
    Complete end-to-end pipeline
    
    Processes:
    1. Extract JD requirements
    2. Parse all resumes
    3. Verify claims for hallucinations
    4. Match resumes to job
    5. Rank candidates with fairness
    """
    try:
        # Initialize services
        extractor = JDExtractor()
        ner = ResumeNERExtractor()
        verifier = ClaimVerifier()
        matcher = JobResumeMatcher()
        ranker = FairnessAwareRanker()
        
        # Step 1: Extract JD
        jd_data = extractor.extract_jd_data(request.jd_text)
        if not isinstance(jd_data, dict):
            jd_data = {"required_skills": [], "required_experience": 0, "required_education": ""}
        
        import hashlib
        job_id = hashlib.md5(request.jd_text.encode()).hexdigest()[:12]
        
        # Step 2-5: Process each resume
        candidates = []
        
        for i, resume_text in enumerate(request.resume_texts):
            try:
                # Parse resume
                resume_data = ner.parse_resume(resume_text)
                if not isinstance(resume_data, dict):
                    resume_data = {}
                
                resume_id = hashlib.md5(resume_text.encode()).hexdigest()[:12]
                
                # Verify claims
                verification = verifier.verify_resume_data(resume_data)
                if not isinstance(verification, dict):
                    verification = {"status": "PENDING", "hallucinations": []}
                
                # Match to job
                match_result = matcher.match_resume_to_job(resume_data, jd_data)
                if not isinstance(match_result, dict):
                    match_result = {"match_score": 0, "skill_match": 0}
                
                candidates.append({
                    "resume_id": resume_id,
                    "candidate_name": resume_data.get("name", f"Candidate {i+1}"),
                    "resume_data": resume_data,
                    "match_data": match_result,
                    "verification": verification,
                    "demographics": {"gender": 1, "race_gender": "Unknown"}
                })
            except Exception as e:
                import logging
                logging.error(f"Error processing resume {i+1}: {str(e)}")
                # Skip this resume but continue with others
                continue
        
        # Step 5: Rank candidates
        ranked = ranker.rank_candidates(
            candidates,
            jd_data,
            use_fairness=request.use_fairness
        )
        
        return {
            "success": True,
            "job_id": job_id,
            "jd_data": jd_data,
            "total_candidates": len(request.resume_texts),
            "ranked_candidates": ranked,
            "fairness_enabled": request.use_fairness
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {str(e)}"
        )