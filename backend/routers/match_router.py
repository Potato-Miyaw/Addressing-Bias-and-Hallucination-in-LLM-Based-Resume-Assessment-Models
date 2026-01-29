"""
Match Router - Endpoints for job-resume matching operations
Handles CRUD operations for matches collection
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from backend.database import (
    save_match,
    get_match,
    get_matches_by_job,
    get_matches_by_resume,
    get_matches_by_tier,
    list_matches,
    delete_match,
    get_match_count
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/matches", tags=["Matches"])


# ==================== Pydantic Models ====================

class SkillMatch(BaseModel):
    """Skill matching details"""
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    additional_skills: List[str] = Field(default_factory=list)
    skill_match_percentage: float = Field(ge=0, le=100)


class RequirementMatch(BaseModel):
    """Requirement matching details"""
    matched_requirements: List[str] = Field(default_factory=list)
    unmatched_requirements: List[str] = Field(default_factory=list)


class CertificationMatch(BaseModel):
    """Certification matching details"""
    matched_certifications: List[str] = Field(default_factory=list)
    missing_certifications: List[str] = Field(default_factory=list)


class MatchScores(BaseModel):
    """Component scores for matching"""
    overall_match_score: float = Field(ge=0, le=1)
    skill_match_score: float = Field(ge=0, le=1)
    experience_match_score: float = Field(ge=0, le=1)
    education_match_score: float = Field(ge=0, le=1)


class MatchCreate(BaseModel):
    """Request model for creating a match"""
    job_id: str
    resume_id: str
    candidate_name: str
    candidate_email: str
    job_title: str
    
    # Scores
    overall_match_score: float = Field(ge=0, le=1)
    skill_match_score: float = Field(ge=0, le=1)
    experience_match_score: float = Field(ge=0, le=1)
    education_match_score: float = Field(ge=0, le=1)
    
    # Match tier: Excellent (≥0.80), Good (0.60-0.79), Fair (0.40-0.59), Poor (<0.40)
    match_tier: str = Field(pattern="^(Excellent|Good|Fair|Poor)$")
    
    # Skills matching
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    additional_skills: List[str] = Field(default_factory=list)
    skill_match_percentage: float = Field(ge=0, le=100)
    
    # Requirements matching
    matched_requirements: List[str] = Field(default_factory=list)
    unmatched_requirements: List[str] = Field(default_factory=list)
    
    # Experience
    candidate_experience_years: Optional[float] = None
    required_experience_years: Optional[float] = None
    
    # Certifications
    matched_certifications: List[str] = Field(default_factory=list)
    missing_certifications: List[str] = Field(default_factory=list)
    
    # Metadata
    matching_algorithm: str = Field(default="hybrid_matcher")
    model_version: Optional[str] = None
    created_by: str = Field(default="system")
    match_source: str = Field(pattern="^(hr_initiated|candidate_initiated)$")
    
    # Additional details
    match_details: Optional[Dict[str, Any]] = None


class MatchResponse(BaseModel):
    """Response model for match data"""
    match_id: str
    job_id: str
    resume_id: str
    candidate_name: str
    candidate_email: str
    job_title: str
    
    overall_match_score: float
    skill_match_score: float
    experience_match_score: float
    education_match_score: float
    match_tier: str
    
    matched_skills: List[str]
    missing_skills: List[str]
    additional_skills: List[str]
    skill_match_percentage: float
    
    matched_requirements: List[str]
    unmatched_requirements: List[str]
    
    candidate_experience_years: Optional[float]
    required_experience_years: Optional[float]
    
    matched_certifications: List[str]
    missing_certifications: List[str]
    
    matching_algorithm: str
    model_version: Optional[str]
    created_at: datetime
    created_by: str
    match_source: str
    match_details: Optional[Dict[str, Any]]


class MatchListResponse(BaseModel):
    """Response model for list of matches"""
    matches: List[MatchResponse]
    total: int
    limit: int
    skip: int


class MatchCountResponse(BaseModel):
    """Response model for match count"""
    total_matches: int


# ==================== Endpoints ====================

@router.post("/create", response_model=MatchResponse, status_code=201)
async def create_match(match_data: MatchCreate):
    """
    Create a new job-resume match
    
    - **job_id**: ID of the job description
    - **resume_id**: ID of the resume
    - **overall_match_score**: Overall matching score (0-1)
    - **match_tier**: Excellent/Good/Fair/Poor
    - **match_source**: hr_initiated or candidate_initiated
    """
    try:
        logger.info(f"Creating match for job {match_data.job_id} and resume {match_data.resume_id}")
        
        # Convert to dict and save
        match_dict = match_data.model_dump()
        result = await save_match(match_dict)
        
        if not result:
            raise HTTPException(status_code=500, detail="Failed to create match")
        
        logger.info(f"✅ Match created: {result['match_id']}")
        return MatchResponse(**result)
    
    except Exception as e:
        logger.error(f"Error creating match: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{match_id}", response_model=MatchResponse)
async def get_match_by_id(match_id: str):
    """
    Get a specific match by ID
    
    - **match_id**: Unique match identifier
    """
    try:
        match = await get_match(match_id)
        
        if not match:
            raise HTTPException(status_code=404, detail=f"Match {match_id} not found")
        
        return MatchResponse(**match)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving match {match_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job/{job_id}", response_model=List[MatchResponse])
async def get_matches_for_job(job_id: str):
    """
    Get all matches for a specific job
    
    Returns matches sorted by overall_match_score (highest first)
    
    - **job_id**: Job description ID
    """
    try:
        matches = await get_matches_by_job(job_id)
        return [MatchResponse(**match) for match in matches]
    
    except Exception as e:
        logger.error(f"Error retrieving matches for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/resume/{resume_id}", response_model=List[MatchResponse])
async def get_matches_for_resume(resume_id: str):
    """
    Get all matches for a specific resume
    
    Returns matches sorted by overall_match_score (highest first)
    
    - **resume_id**: Resume ID
    """
    try:
        matches = await get_matches_by_resume(resume_id)
        return [MatchResponse(**match) for match in matches]
    
    except Exception as e:
        logger.error(f"Error retrieving matches for resume {resume_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/job/{job_id}/tier/{tier}", response_model=List[MatchResponse])
async def get_matches_by_tier_for_job(
    job_id: str,
    tier: str
):
    """
    Get matches for a job filtered by tier
    
    - **job_id**: Job description ID
    - **tier**: Match tier (Excellent/Good/Fair/Poor)
    """
    valid_tiers = ["Excellent", "Good", "Fair", "Poor"]
    if tier not in valid_tiers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tier. Must be one of: {', '.join(valid_tiers)}"
        )
    
    try:
        matches = await get_matches_by_tier(job_id, tier)
        return [MatchResponse(**match) for match in matches]
    
    except Exception as e:
        logger.error(f"Error retrieving {tier} matches for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=MatchListResponse)
async def list_all_matches(
    limit: int = Query(50, ge=1, le=1000, description="Number of matches to return"),
    skip: int = Query(0, ge=0, description="Number of matches to skip")
):
    """
    List all matches with pagination
    
    Returns matches sorted by created_at (newest first)
    
    - **limit**: Maximum number of matches to return (1-1000)
    - **skip**: Number of matches to skip (for pagination)
    """
    try:
        matches = await list_matches(limit=limit, skip=skip)
        total = await get_match_count()
        
        return MatchListResponse(
            matches=[MatchResponse(**match) for match in matches],
            total=total,
            limit=limit,
            skip=skip
        )
    
    except Exception as e:
        logger.error(f"Error listing matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/count", response_model=MatchCountResponse)
async def count_matches():
    """
    Get total count of matches in the database
    """
    try:
        count = await get_match_count()
        return MatchCountResponse(total_matches=count)
    
    except Exception as e:
        logger.error(f"Error counting matches: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{match_id}", status_code=204)
async def delete_match_by_id(match_id: str):
    """
    Delete a match by ID
    
    - **match_id**: Unique match identifier
    """
    try:
        success = await delete_match(match_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Match {match_id} not found")
        
        logger.info(f"✅ Match deleted: {match_id}")
        return None
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting match {match_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Helper Endpoint ====================

@router.get("/job/{job_id}/summary")
async def get_job_match_summary(job_id: str):
    """
    Get summary statistics for all matches of a job
    
    Returns:
    - Total matches
    - Count by tier (Excellent/Good/Fair/Poor)
    - Average score
    """
    try:
        matches = await get_matches_by_job(job_id)
        
        if not matches:
            return {
                "job_id": job_id,
                "total_matches": 0,
                "tier_breakdown": {
                    "Excellent": 0,
                    "Good": 0,
                    "Fair": 0,
                    "Poor": 0
                },
                "average_score": 0.0
            }
        
        tier_counts = {"Excellent": 0, "Good": 0, "Fair": 0, "Poor": 0}
        total_score = 0.0
        
        for match in matches:
            tier = match.get("match_tier", "Poor")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            total_score += match.get("overall_match_score", 0.0)
        
        return {
            "job_id": job_id,
            "total_matches": len(matches),
            "tier_breakdown": tier_counts,
            "average_score": round(total_score / len(matches), 3)
        }
    
    except Exception as e:
        logger.error(f"Error generating match summary for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
