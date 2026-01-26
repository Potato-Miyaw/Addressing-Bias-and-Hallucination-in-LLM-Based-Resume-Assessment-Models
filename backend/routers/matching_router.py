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

router = APIRouter(prefix="/api/match", tags=["Matching"])

# Lazy load services
job_matcher = None
ner_extractor = None

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

# Pydantic models
class MatchRequest(BaseModel):
    resume_id: str
    job_id: str
    resume_data: Dict[str, Any]
    jd_data: Dict[str, Any]

@router.post("/")
async def match_resume_to_job(request: MatchRequest):
    """Match a resume to a job description"""
    try:
        matcher = get_job_matcher()
        extractor = get_ner_extractor()
        
        resume_data = request.resume_data
        
        # Check if we need to run NER extraction
        needs_extraction = (
            'text' in resume_data and 
            'skills' not in resume_data and 
            'primary_skills' not in resume_data
        )
        
        if needs_extraction:
            print(f"Running NER extraction for resume {request.resume_id}")
            resume_data = extractor.parse_resume(resume_data['text'])
            
            # 🔍 DEBUG: Print extracted skills
            print(f"DEBUG - Resume {request.resume_id}:")
            print(f"  Primary skills: {resume_data.get('primary_skills', [])}")
            print(f"  Secondary skills: {resume_data.get('secondary_skills', [])}")
            print(f"  Education: {resume_data.get('education', [])}")
            print(f"  Experience: {resume_data.get('total_experience_(months)', 0)} months")
        
        # 🔍 DEBUG: Print JD skills
        print(f"DEBUG - JD skills: {request.jd_data.get('required_skills', [])}")
        
        # Now match
        match_result = matcher.match_resume_to_job(resume_data, request.jd_data)
        
        # 🔍 DEBUG: Print match result
        print(f"DEBUG - Match result: {match_result}")
        
        return {
            "success": True,
            "resume_id": request.resume_id,
            "job_id": request.job_id,
            "match_result": match_result,
            "ner_extracted": needs_extraction
        }
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
    resumes: List[Dict[str, Any]]
):
    """Match multiple resumes to a single job"""
    matcher = get_job_matcher()
    extractor = get_ner_extractor()
    results = []
    
    for resume in resumes:
        try:
            resume_data = resume["resume_data"]
            
            # Check if NER extraction needed
            if 'text' in resume_data and 'skills' not in resume_data and 'primary_skills' not in resume_data:
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
        "results": results
    }