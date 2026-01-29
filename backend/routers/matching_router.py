"""
Job-Resume Matching Router
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any, List
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature4_matcher import JobResumeMatcher
from backend.services.feature2_bert_ner import ResumeNERExtractor
from backend.services.feature2_hybrid_ner import HybridResumeNERExtractor
from backend.database import get_job_description

router = APIRouter(prefix="/api/match", tags=["Matching"])

# Lazy load services
job_matcher = None
ner_extractor = None
hybrid_extractor = None

def get_job_matcher():
    global job_matcher
    if job_matcher is None:
        job_matcher = JobResumeMatcher()
    return job_matcher

def get_ner_extractor():
    global ner_extractor
    if ner_extractor is None:
        ner_extractor = ResumeNERExtractor()
    return ner_extractor

def get_hybrid_extractor():
    global hybrid_extractor
    if hybrid_extractor is None:
        hybrid_extractor = HybridResumeNERExtractor()
    return hybrid_extractor

# Pydantic models
class MatchRequest(BaseModel):
    resume_id: str
    job_id: str
    resume_data: Dict[str, Any]
    jd_data: Dict[str, Any]
    save_to_db: bool = True

class MatchWithJobIdRequest(BaseModel):
    """Match using job_id from database"""
    resume_id: str
    job_id: str
    resume_data: Dict[str, Any]
    save_to_db: bool = True

@router.post("/")
async def match_resume_to_job(request: MatchRequest):
    """
    Match a resume to a job description
    Runs full NER extraction if resume only has basic info
    """
    try:
        matcher = get_job_matcher()
        extractor = get_hybrid_extractor()  # Use HYBRID for best results!
        
        resume_data = request.resume_data
        
        # Check if we need to run NER extraction
        needs_extraction = (
            'text' in resume_data and 
            'skills' not in resume_data and 
            'skills' not in resume_data
        )
        
        if needs_extraction:
            print(f"Running HYBRID NER extraction for resume {request.resume_id}")
            resume_data = extractor.parse_resume(resume_data['text'])
        
        # Now match
        match_result = matcher.match_resume_to_job(resume_data, request.jd_data)
        
        return {
            "success": True,
            "resume_id": request.resume_id,
            "job_id": request.job_id,
            "match_result": match_result,
            "ner_extracted": needs_extraction,
            "saved_to_db": False
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching failed: {str(e)}"
        )

@router.post("/with-job-id")
async def match_with_job_id(request: MatchWithJobIdRequest):
    """
    Match a resume to a job using job_id from database
    Fetches job description from MongoDB automatically
    """
    try:
        # Fetch job description from database
        jd_data = await get_job_description(request.job_id)
        if not jd_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job not found in database: {request.job_id}"
            )
        
        matcher = get_job_matcher()
        extractor = get_hybrid_extractor()
        
        resume_data = request.resume_data
        
        # Check if we need to run NER extraction
        needs_extraction = (
            'text' in resume_data and 
            'skills' not in resume_data
        )
        
        if needs_extraction:
            print(f"Running HYBRID NER extraction for resume {request.resume_id}")
            resume_data = extractor.parse_resume(resume_data['text'])
        
        # Prepare JD data for matching
        jd_match_data = {
            "required_skills": jd_data.get('required_skills', []),
            "required_experience": jd_data.get('required_experience', 0),
            "required_education": jd_data.get('required_education', '')
        }
        
        # Now match
        match_result = matcher.match_resume_to_job(resume_data, jd_match_data)
        
        return {
            "success": True,
            "resume_id": request.resume_id,
            "job_id": request.job_id,
            "job_title": jd_data.get('job_title', 'Unknown'),
            "match_result": match_result,
            "ner_extracted": needs_extraction,
            "saved_to_db": False
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Matching failed: {str(e)}"
        )

@router.post("/batch")
async def batch_match_resumes(
    job_id: str,
    jd_data: Dict[str, Any],
    resumes: List[Dict[str, Any]],
    save_to_db: bool = True
):
    """Match multiple resumes to a single job"""
    matcher = get_job_matcher()
    extractor = get_hybrid_extractor()  # Use HYBRID for best results!
    results = []
    
    for resume in resumes:
        try:
            resume_data = resume["resume_data"]
            
            # Check if NER extraction needed
            if 'text' in resume_data and 'skills' not in resume_data:
                resume_data = extractor.parse_resume(resume_data['text'])
            
            match_result = matcher.match_resume_to_job(resume_data, jd_data)
            
            results.append({
                "resume_id": resume.get("resume_id"),
                "match_result": match_result,
                "status": "SUCCESS"
            })
        except Exception as e:
            results.append({
                "resume_id": resume.get("resume_id"),
                "status": "FAILED",
                "error": str(e)
            })
    
    return {
        "job_id": job_id,
        "total_resumes": len(resumes),
        "successful_matches": len([r for r in results if r["status"] == "SUCCESS"]),
        "results": results,
        "saved_to_db": False
    }