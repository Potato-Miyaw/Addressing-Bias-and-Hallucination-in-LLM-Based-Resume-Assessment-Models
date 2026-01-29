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
from backend.database import get_job_description, save_match
from backend.utils.id_generator import generate_match_id

router = APIRouter(prefix="/api/match", tags=["Matching"])

# Helper function to determine match tier
def calculate_match_tier(score: float) -> str:
    """Calculate match tier based on overall score"""
    if score >= 0.80:
        return "Excellent"
    elif score >= 0.60:
        return "Good"
    elif score >= 0.40:
        return "Fair"
    else:
        return "Poor"

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
    match_source: str = "hr_initiated"  # hr_initiated or candidate_initiated

class MatchWithJobIdRequest(BaseModel):
    """Match using job_id from database"""
    resume_id: str
    job_id: str
    resume_data: Dict[str, Any]
    save_to_db: bool = True
    match_source: str = "hr_initiated"  # hr_initiated or candidate_initiated

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
        
        # Save to database if requested
        match_id = None
        if request.save_to_db:
            try:
                match_id = generate_match_id(request.job_id, request.resume_id)
                match_data = {
                    "match_id": match_id,
                    "job_id": request.job_id,
                    "resume_id": request.resume_id,
                    "candidate_name": resume_data.get("name", "Unknown"),
                    "candidate_email": resume_data.get("email", ""),
                    "job_title": request.jd_data.get("job_title", "Unknown"),
                    "overall_match_score": match_result.get("match_score", 0.0) / 100.0,  # Convert % to 0-1
                    "skill_match_score": match_result.get("skill_match", 0.0) / 100.0,
                    "experience_match_score": match_result.get("experience_match", 0.0) / 100.0,
                    "education_match_score": match_result.get("education_match", 0.0) / 100.0,
                    "match_tier": calculate_match_tier(match_result.get("match_score", 0.0) / 100.0),
                    "matched_skills": match_result.get("matched_skills", []),
                    "missing_skills": match_result.get("skill_gaps", []),  # Matcher uses 'skill_gaps'
                    "additional_skills": match_result.get("additional_skills", []),
                    "skill_match_percentage": match_result.get("skill_match", 0.0),
                    "matched_requirements": match_result.get("matched_requirements", []),
                    "unmatched_requirements": match_result.get("unmatched_requirements", []),
                    "candidate_experience_years": resume_data.get("total_experience_years"),
                    "required_experience_years": request.jd_data.get("required_experience"),
                    "matched_certifications": match_result.get("matched_certifications", []),
                    "missing_certifications": match_result.get("missing_certifications", []),
                    "matching_algorithm": "hybrid_matcher",
                    "model_version": "v1.0",
                    "created_by": "system",
                    "match_source": request.match_source,
                    "match_details": match_result
                }
                await save_match(match_data)
            except Exception as e:
                print(f"Failed to save match to database: {e}")
        
        return {
            "success": True,
            "resume_id": request.resume_id,
            "job_id": request.job_id,
            "match_id": match_id,
            "match_result": match_result,
            "ner_extracted": needs_extraction,
            "saved_to_db": match_id is not None
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
        
        # Save to database if requested
        match_id = None
        if request.save_to_db:
            try:
                match_id = generate_match_id(request.job_id, request.resume_id)
                match_data = {
                    "match_id": match_id,
                    "job_id": request.job_id,
                    "resume_id": request.resume_id,
                    "candidate_name": resume_data.get("name", "Unknown"),
                    "candidate_email": resume_data.get("email", ""),
                    "job_title": jd_data.get("job_title", "Unknown"),
                    "overall_match_score": match_result.get("match_score", 0.0) / 100.0,  # Convert % to 0-1
                    "skill_match_score": match_result.get("skill_match", 0.0) / 100.0,
                    "experience_match_score": match_result.get("experience_match", 0.0) / 100.0,
                    "education_match_score": match_result.get("education_match", 0.0) / 100.0,
                    "match_tier": calculate_match_tier(match_result.get("match_score", 0.0) / 100.0),
                    "matched_skills": match_result.get("matched_skills", []),
                    "missing_skills": match_result.get("skill_gaps", []),  # Matcher uses 'skill_gaps'
                    "additional_skills": match_result.get("additional_skills", []),
                    "skill_match_percentage": match_result.get("skill_match", 0.0),
                    "matched_requirements": match_result.get("matched_requirements", []),
                    "unmatched_requirements": match_result.get("unmatched_requirements", []),
                    "candidate_experience_years": resume_data.get("total_experience_years"),
                    "required_experience_years": jd_data.get("required_experience"),
                    "matched_certifications": match_result.get("matched_certifications", []),
                    "missing_certifications": match_result.get("missing_certifications", []),
                    "matching_algorithm": "hybrid_matcher",
                    "model_version": "v1.0",
                    "created_by": "system",
                    "match_source": request.match_source,
                    "match_details": match_result
                }
                await save_match(match_data)
            except Exception as e:
                print(f"Failed to save match to database: {e}")
        
        return {
            "success": True,
            "resume_id": request.resume_id,
            "job_id": request.job_id,
            "match_id": match_id,
            "job_title": jd_data.get('job_title', 'Unknown'),
            "match_result": match_result,
            "ner_extracted": needs_extraction,
            "saved_to_db": match_id is not None
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