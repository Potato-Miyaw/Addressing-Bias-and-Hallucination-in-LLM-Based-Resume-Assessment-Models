"""
Data Query Router - Query and retrieve stored data from MongoDB
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.database import (
    get_job_description,
    list_job_descriptions,
    search_jobs_by_title,
    get_job_count,
    get_ground_truth,
    list_ground_truths,
    get_ground_truth_count
)

router = APIRouter(prefix="/api/data", tags=["Data Queries"])

# ============================================================================
# Job Description Queries
# ============================================================================

@router.get("/jobs")
async def query_jobs(limit: int = Query(50, ge=1, le=100), skip: int = Query(0, ge=0)):
    """List all job descriptions with pagination"""
    jobs = await list_job_descriptions(limit=limit, skip=skip)
    total = await get_job_count()
    
    return {
        "success": True,
        "total": total,
        "count": len(jobs),
        "limit": limit,
        "skip": skip,
        "jobs": jobs
    }

@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get specific job description by ID"""
    job = await get_job_description(job_id)
    if not job:
        raise HTTPException(404, f"Job not found: {job_id}")
    return {
        "success": True,
        "job": job
    }

@router.get("/jobs/search/title")
async def search_jobs(title: str = Query(..., min_length=1)):
    """Search job descriptions by title (case-insensitive)"""
    jobs = await search_jobs_by_title(title)
    return {
        "success": True,
        "search_term": title,
        "count": len(jobs),
        "jobs": jobs
    }

@router.get("/statistics")
async def get_statistics():
    """Get database statistics"""
    job_count = await get_job_count()
    resume_count = await get_ground_truth_count()
    
    return {
        "success": True,
        "statistics": {
            "total_jobs": job_count,
            "total_resumes": resume_count,
            "total_matches": 0,
            "total_verifications": 0
        }
    }


# ============================================================================
# Ground Truth (Raw Resume) Queries
# ============================================================================

@router.get("/resumes")
async def query_resumes(limit: int = Query(50, ge=1, le=100), skip: int = Query(0, ge=0)):
    """List all ground truth (raw resume text) entries with pagination"""
    resumes = await list_ground_truths(limit=limit, skip=skip)
    total = await get_ground_truth_count()
    
    return {
        "success": True,
        "total": total,
        "count": len(resumes),
        "limit": limit,
        "skip": skip,
        "resumes": resumes
    }

@router.get("/resumes/{resume_id}")
async def get_resume(resume_id: str):
    """Get specific ground truth (raw resume) by ID"""
    resume = await get_ground_truth(resume_id)
    if not resume:
        raise HTTPException(404, f"Resume not found: {resume_id}")
    return {
        "success": True,
        "resume": resume
    }
