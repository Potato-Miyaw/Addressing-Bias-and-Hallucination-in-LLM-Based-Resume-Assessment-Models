"""
ID Generation Utility - Hybrid Strategy
Generates unique IDs with content hash for duplicate detection
"""

import hashlib
from datetime import datetime
from typing import Tuple

def generate_job_id(jd_text: str, job_title: str = None) -> Tuple[str, str]:
    """
    Generate unique job ID with duplicate detection
    
    Args:
        jd_text: Job description text
        job_title: Optional job title for readable ID
    
    Returns:
        Tuple of (unique_id, content_hash)
        - unique_id: "job_20260128151530123_a3f2e1b9"
        - content_hash: "a3f2e1b9c8d7f6e5" (for duplicate detection)
    """
    # Generate content hash (SHA-256 for better collision resistance)
    content_hash = hashlib.sha256(jd_text.encode()).hexdigest()[:16]
    
    # Generate timestamp with microseconds for uniqueness
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17]
    
    # Create readable prefix
    prefix = "job"
    if job_title:
        # Extract first word of job title, make safe for ID
        safe_title = "".join(c for c in job_title.lower().split()[0] if c.isalnum())[:8]
        if safe_title:
            prefix = f"job_{safe_title}"
    
    # Combine: job_20260128151530123_a3f2e1b9
    unique_id = f"{prefix}_{timestamp}_{content_hash[:8]}"
    
    return unique_id, content_hash


def generate_resume_id(resume_text: str, candidate_name: str = None) -> Tuple[str, str]:
    """
    Generate unique resume ID with duplicate detection
    
    Args:
        resume_text: Resume text content
        candidate_name: Optional candidate name for readable ID
    
    Returns:
        Tuple of (unique_id, content_hash)
        - unique_id: "res_johndoe_20260128151530123_b4c3d2e1"
        - content_hash: "b4c3d2e1f0a9b8c7" (for duplicate detection)
    """
    # Generate content hash
    content_hash = hashlib.sha256(resume_text.encode()).hexdigest()[:16]
    
    # Generate timestamp with microseconds
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17]
    
    # Create readable prefix
    prefix = "res"
    if candidate_name:
        # Extract and clean name for ID
        safe_name = "".join(c for c in candidate_name.lower() if c.isalnum())[:10]
        if safe_name:
            prefix = f"res_{safe_name}"
    
    # Combine: res_johndoe_20260128151530123_b4c3d2e1
    unique_id = f"{prefix}_{timestamp}_{content_hash[:8]}"
    
    return unique_id, content_hash


def generate_match_id(job_id: str, resume_id: str) -> str:
    """
    Generate match ID from job and resume IDs
    
    Args:
        job_id: Job ID
        resume_id: Resume ID
    
    Returns:
        Match ID: "match_{job_id}_{resume_id}"
    """
    return f"match_{job_id}_{resume_id}"


def generate_verification_id(resume_id: str) -> str:
    """
    Generate verification ID for a resume
    
    Args:
        resume_id: Resume ID
    
    Returns:
        Verification ID with timestamp
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17]
    return f"ver_{resume_id}_{timestamp}"


def generate_ranking_id(job_id: str) -> str:
    """
    Generate ranking ID for a job
    
    Args:
        job_id: Job ID
    
    Returns:
        Ranking ID with timestamp
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17]
    return f"rank_{job_id}_{timestamp}"


def generate_feedback_id(resume_id: str, field: str) -> str:
    """
    Generate feedback ID
    
    Args:
        resume_id: Resume ID
        field: Field being corrected (name, email, phone, etc.)
    
    Returns:
        Feedback ID with timestamp
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17]
    return f"fb_{field}_{resume_id}_{timestamp}"


def generate_experiment_id() -> str:
    """
    Generate bias experiment ID
    
    Returns:
        Experiment ID with timestamp
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17]
    return f"exp_{timestamp}"


def generate_pipeline_run_id(job_id: str) -> str:
    """
    Generate pipeline run ID
    
    Args:
        job_id: Job ID
    
    Returns:
        Pipeline run ID with timestamp
    """
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")[:17]
    return f"pipe_{job_id}_{timestamp}"
