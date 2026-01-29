"""
Database module - MongoDB integration for persistent storage
"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING, DESCENDING
from typing import Dict, Any, List, Optional
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# MongoDB connection settings
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DATABASE_NAME = os.getenv("DATABASE_NAME", "resume_screening_db")

class Database:
    """MongoDB database singleton"""
    client: AsyncIOMotorClient = None
    db = None

db_instance = Database()

async def connect_to_mongo():
    """Initialize MongoDB connection"""
    try:
        db_instance.client = AsyncIOMotorClient(MONGODB_URL)
        db_instance.db = db_instance.client[DATABASE_NAME]
        
        # Test connection
        await db_instance.client.admin.command('ping')
        logger.info(f"✅ Connected to MongoDB: {DATABASE_NAME}")
        
        # Create indexes
        await create_indexes()
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close MongoDB connection"""
    if db_instance.client:
        db_instance.client.close()
        logger.info("MongoDB connection closed")

async def create_indexes():
    """Create database indexes for better performance"""
    try:
        # Job descriptions indexes
        await db_instance.db.job_descriptions.create_index([("job_id", ASCENDING)], unique=True)
        await db_instance.db.job_descriptions.create_index([("content_hash", ASCENDING)])
        await db_instance.db.job_descriptions.create_index([("created_at", DESCENDING)])
        await db_instance.db.job_descriptions.create_index([("job_title", ASCENDING)])
        
        # Ground truth (raw resume text) indexes
        await db_instance.db.ground_truth.create_index([("resume_id", ASCENDING)], unique=True)
        await db_instance.db.ground_truth.create_index([("content_hash", ASCENDING)])
        await db_instance.db.ground_truth.create_index([("uploaded_at", DESCENDING)])
        await db_instance.db.ground_truth.create_index([("file_type", ASCENDING)])
        await db_instance.db.ground_truth.create_index([("candidate_email", ASCENDING)])
        await db_instance.db.ground_truth.create_index([("upload_source", ASCENDING)])
        
        # Matches indexes
        await db_instance.db.matches.create_index([("match_id", ASCENDING)], unique=True)
        await db_instance.db.matches.create_index([("job_id", ASCENDING)])
        await db_instance.db.matches.create_index([("resume_id", ASCENDING)])
        await db_instance.db.matches.create_index([("match_tier", ASCENDING)])
        await db_instance.db.matches.create_index([("overall_match_score", DESCENDING)])
        await db_instance.db.matches.create_index([("created_at", DESCENDING)])
        await db_instance.db.matches.create_index([("candidate_email", ASCENDING)])
        await db_instance.db.matches.create_index([("match_source", ASCENDING)])
        
        # Questionnaires indexes
        await db_instance.db.questionnaires.create_index([("questionnaire_id", ASCENDING)], unique=True)
        await db_instance.db.questionnaires.create_index([("created_at", DESCENDING)])
        await db_instance.db.questionnaires.create_index([("status", ASCENDING)])
        
        # Questionnaire invitations indexes
        await db_instance.db.questionnaire_invitations.create_index([("invitation_id", ASCENDING)], unique=True)
        await db_instance.db.questionnaire_invitations.create_index([("token", ASCENDING)], unique=True)
        await db_instance.db.questionnaire_invitations.create_index([("candidate_email", ASCENDING)])
        await db_instance.db.questionnaire_invitations.create_index([("questionnaire_id", ASCENDING)])
        await db_instance.db.questionnaire_invitations.create_index([("used", ASCENDING)])
        await db_instance.db.questionnaire_invitations.create_index([("expires_at", ASCENDING)])
        
        # Questionnaire responses indexes
        await db_instance.db.questionnaire_responses.create_index([("response_id", ASCENDING)], unique=True)
        await db_instance.db.questionnaire_responses.create_index([("token", ASCENDING)])
        await db_instance.db.questionnaire_responses.create_index([("candidate_email", ASCENDING)])
        await db_instance.db.questionnaire_responses.create_index([("questionnaire_id", ASCENDING)])
        await db_instance.db.questionnaire_responses.create_index([("submitted_at", DESCENDING)])
        
        logger.info("✅ Database indexes created for all collections")
    except Exception as e:
        logger.warning(f"Index creation warning: {e}")

def get_database():
    """Get database instance"""
    return db_instance.db

# ============================================================================
# CRUD Operations for Job Descriptions
# ============================================================================

async def save_job_description(job_data: Dict[str, Any]) -> bool:
    """
    Save or update job description
    
    Args:
        job_data: Job description data with job_id and content_hash
    
    Returns:
        True if successful, False otherwise
    """
    try:
        job_data["updated_at"] = datetime.utcnow()
        if "created_at" not in job_data:
            job_data["created_at"] = datetime.utcnow()
        
        await db_instance.db.job_descriptions.update_one(
            {"job_id": job_data["job_id"]},
            {"$set": job_data},
            upsert=True
        )
        logger.info(f"✅ Saved job description: {job_data['job_id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save job description: {e}")
        return False

async def get_job_description(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Get job description by ID
    
    Args:
        job_id: Job ID
    
    Returns:
        Job description dict or None if not found
    """
    try:
        job = await db_instance.db.job_descriptions.find_one({"job_id": job_id})
        if job:
            job.pop("_id", None)  # Remove MongoDB internal ID
        return job
    except Exception as e:
        logger.error(f"❌ Failed to get job description: {e}")
        return None

async def get_job_by_content_hash(content_hash: str) -> Optional[Dict[str, Any]]:
    """
    Check if job with same content already exists (duplicate detection)
    
    Args:
        content_hash: Content hash of job description
    
    Returns:
        Existing job description or None
    """
    try:
        job = await db_instance.db.job_descriptions.find_one({"content_hash": content_hash})
        if job:
            job.pop("_id", None)
        return job
    except Exception as e:
        logger.error(f"❌ Failed to check duplicate job: {e}")
        return None

async def list_job_descriptions(limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """
    List all job descriptions
    
    Args:
        limit: Maximum number of jobs to return
        skip: Number of jobs to skip
    
    Returns:
        List of job descriptions
    """
    try:
        cursor = db_instance.db.job_descriptions.find().sort("created_at", DESCENDING).skip(skip).limit(limit)
        jobs = []
        async for job in cursor:
            job.pop("_id", None)
            jobs.append(job)
        return jobs
    except Exception as e:
        logger.error(f"❌ Failed to list job descriptions: {e}")
        return []

async def search_jobs_by_title(title: str) -> List[Dict[str, Any]]:
    """
    Search job descriptions by title
    
    Args:
        title: Job title search term (case-insensitive)
    
    Returns:
        List of matching job descriptions
    """
    try:
        cursor = db_instance.db.job_descriptions.find(
            {"job_title": {"$regex": title, "$options": "i"}}
        ).sort("created_at", DESCENDING)
        
        jobs = []
        async for job in cursor:
            job.pop("_id", None)
            jobs.append(job)
        return jobs
    except Exception as e:
        logger.error(f"❌ Failed to search jobs: {e}")
        return []

async def delete_job_description(job_id: str) -> bool:
    """
    Delete a job description
    
    Args:
        job_id: Job ID
    
    Returns:
        True if deleted, False otherwise
    """
    try:
        result = await db_instance.db.job_descriptions.delete_one({"job_id": job_id})
        if result.deleted_count > 0:
            logger.info(f"✅ Deleted job description: {job_id}")
            return True
        else:
            logger.warning(f"⚠️ Job not found: {job_id}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to delete job description: {e}")
        return False

async def get_job_count() -> int:
    """
    Get total count of job descriptions
    
    Returns:
        Total number of jobs
    """
    try:
        count = await db_instance.db.job_descriptions.count_documents({})
        return count
    except Exception as e:
        logger.error(f"❌ Failed to count jobs: {e}")
        return 0


# ============================================================================
# CRUD Operations for Ground Truth (Raw Resume Text)
# ============================================================================

async def save_ground_truth(ground_truth_data: Dict[str, Any]) -> bool:
    """
    Save or update raw resume text (ground truth)
    
    Args:
        ground_truth_data: Dict with resume_id, content_hash, raw_text, etc.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        ground_truth_data["updated_at"] = datetime.utcnow()
        if "uploaded_at" not in ground_truth_data:
            ground_truth_data["uploaded_at"] = datetime.utcnow()
        
        await db_instance.db.ground_truth.update_one(
            {"resume_id": ground_truth_data["resume_id"]},
            {"$set": ground_truth_data},
            upsert=True
        )
        logger.info(f"✅ Saved ground truth: {ground_truth_data['resume_id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save ground truth: {e}")
        return False

async def get_ground_truth(resume_id: str) -> Optional[Dict[str, Any]]:
    """
    Get raw resume text by resume_id
    
    Args:
        resume_id: Resume ID
    
    Returns:
        Ground truth dict or None if not found
    """
    try:
        ground_truth = await db_instance.db.ground_truth.find_one({"resume_id": resume_id})
        if ground_truth:
            ground_truth.pop("_id", None)
        return ground_truth
    except Exception as e:
        logger.error(f"❌ Failed to get ground truth: {e}")
        return None

async def get_ground_truth_by_content_hash(content_hash: str) -> Optional[Dict[str, Any]]:
    """
    Check if resume with same content already exists (duplicate detection)
    
    Args:
        content_hash: Content hash of resume text
    
    Returns:
        Existing ground truth or None
    """
    try:
        ground_truth = await db_instance.db.ground_truth.find_one({"content_hash": content_hash})
        if ground_truth:
            ground_truth.pop("_id", None)
        return ground_truth
    except Exception as e:
        logger.error(f"❌ Failed to check duplicate resume: {e}")
        return None

async def list_ground_truths(limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """
    List all ground truth entries
    
    Args:
        limit: Maximum number of entries to return
        skip: Number of entries to skip
    
    Returns:
        List of ground truth entries
    """
    try:
        cursor = db_instance.db.ground_truth.find().sort("uploaded_at", DESCENDING).skip(skip).limit(limit)
        ground_truths = []
        async for gt in cursor:
            gt.pop("_id", None)
            ground_truths.append(gt)
        return ground_truths
    except Exception as e:
        logger.error(f"❌ Failed to list ground truths: {e}")
        return []

async def delete_ground_truth(resume_id: str) -> bool:
    """
    Delete a ground truth entry
    
    Args:
        resume_id: Resume ID
    
    Returns:
        True if deleted, False otherwise
    """
    try:
        result = await db_instance.db.ground_truth.delete_one({"resume_id": resume_id})
        if result.deleted_count > 0:
            logger.info(f"✅ Deleted ground truth: {resume_id}")
            return True
        else:
            logger.warning(f"⚠️ Ground truth not found: {resume_id}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to delete ground truth: {e}")
        return False

async def get_ground_truth_count() -> int:
    """
    Get total count of ground truth entries
    
    Returns:
        Total number of ground truth entries
    """
    try:
        count = await db_instance.db.ground_truth.count_documents({})
        return count
    except Exception as e:
        logger.error(f"❌ Failed to count ground truths: {e}")
        return 0

# ============================================================================
# CRUD Operations for Matches (Job-Resume Matching Results)
# ============================================================================

async def save_match(match_data: Dict[str, Any]) -> bool:
    """
    Save job-resume match result
    
    Args:
        match_data: Dict with match_id, job_id, resume_id, scores, etc.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        match_data["created_at"] = datetime.utcnow()
        
        await db_instance.db.matches.update_one(
            {"match_id": match_data["match_id"]},
            {"$set": match_data},
            upsert=True
        )
        logger.info(f"✅ Saved match: {match_data['match_id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save match: {e}")
        return False

async def get_match(match_id: str) -> Optional[Dict[str, Any]]:
    """
    Get match by ID
    
    Args:
        match_id: Match ID
    
    Returns:
        Match data or None
    """
    try:
        match = await db_instance.db.matches.find_one({"match_id": match_id})
        if match:
            match.pop("_id", None)
            return match
        return None
    except Exception as e:
        logger.error(f"❌ Failed to get match: {e}")
        return None

async def get_matches_by_job(job_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Get all matches for a specific job
    
    Args:
        job_id: Job ID
        limit: Maximum number of matches to return
        skip: Number of matches to skip
    
    Returns:
        List of matches sorted by overall_match_score descending
    """
    try:
        cursor = db_instance.db.matches.find({"job_id": job_id})\
            .sort("overall_match_score", DESCENDING)\
            .skip(skip)\
            .limit(limit)
        
        matches = []
        async for match in cursor:
            match.pop("_id", None)
            matches.append(match)
        return matches
    except Exception as e:
        logger.error(f"❌ Failed to get matches by job: {e}")
        return []

async def get_matches_by_resume(resume_id: str, limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """
    Get all matches for a specific resume
    
    Args:
        resume_id: Resume ID
        limit: Maximum number of matches to return
        skip: Number of matches to skip
    
    Returns:
        List of matches sorted by overall_match_score descending
    """
    try:
        cursor = db_instance.db.matches.find({"resume_id": resume_id})\
            .sort("overall_match_score", DESCENDING)\
            .skip(skip)\
            .limit(limit)
        
        matches = []
        async for match in cursor:
            match.pop("_id", None)
            matches.append(match)
        return matches
    except Exception as e:
        logger.error(f"❌ Failed to get matches by resume: {e}")
        return []

async def get_matches_by_tier(job_id: str, tier: str) -> List[Dict[str, Any]]:
    """
    Get matches filtered by match tier
    
    Args:
        job_id: Job ID
        tier: Match tier (Excellent/Good/Fair/Poor)
    
    Returns:
        List of matches with specified tier
    """
    try:
        cursor = db_instance.db.matches.find({
            "job_id": job_id,
            "match_tier": tier
        }).sort("overall_match_score", DESCENDING)
        
        matches = []
        async for match in cursor:
            match.pop("_id", None)
            matches.append(match)
        return matches
    except Exception as e:
        logger.error(f"❌ Failed to get matches by tier: {e}")
        return []

async def list_matches(limit: int = 50, skip: int = 0) -> List[Dict[str, Any]]:
    """
    List all matches with pagination
    
    Args:
        limit: Maximum number of matches to return
        skip: Number of matches to skip
    
    Returns:
        List of matches sorted by created_at descending
    """
    try:
        cursor = db_instance.db.matches.find()\
            .sort("created_at", DESCENDING)\
            .skip(skip)\
            .limit(limit)
        
        matches = []
        async for match in cursor:
            match.pop("_id", None)
            matches.append(match)
        return matches
    except Exception as e:
        logger.error(f"❌ Failed to list matches: {e}")
        return []

async def delete_match(match_id: str) -> bool:
    """
    Delete a match
    
    Args:
        match_id: Match ID
    
    Returns:
        True if deleted, False otherwise
    """
    try:
        result = await db_instance.db.matches.delete_one({"match_id": match_id})
        if result.deleted_count > 0:
            logger.info(f"✅ Deleted match: {match_id}")
            return True
        else:
            logger.warning(f"⚠️ Match not found: {match_id}")
            return False
    except Exception as e:
        logger.error(f"❌ Failed to delete match: {e}")
        return False

async def get_match_count() -> int:
    """
    Get total count of matches
    
    Returns:
        Total number of matches
    """
    try:
        count = await db_instance.db.matches.count_documents({})
        return count
    except Exception as e:
        logger.error(f"❌ Failed to count matches: {e}")
        return 0


# ============================================================================
# CRUD Operations for Questionnaires
# ============================================================================

async def save_questionnaire(questionnaire_data: Dict[str, Any]) -> bool:
    """
    Save or update questionnaire
    
    Args:
        questionnaire_data: Questionnaire data with questionnaire_id
    
    Returns:
        True if successful, False otherwise
    """
    try:
        questionnaire_data["updated_at"] = datetime.utcnow()
        if "created_at" not in questionnaire_data:
            questionnaire_data["created_at"] = datetime.utcnow()
        
        await db_instance.db.questionnaires.update_one(
            {"questionnaire_id": questionnaire_data["questionnaire_id"]},
            {"$set": questionnaire_data},
            upsert=True
        )
        logger.info(f"✅ Saved questionnaire: {questionnaire_data['questionnaire_id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save questionnaire: {e}")
        return False


async def get_questionnaire(questionnaire_id: str) -> Optional[Dict[str, Any]]:
    """
    Get questionnaire by ID
    
    Args:
        questionnaire_id: Questionnaire ID
    
    Returns:
        Questionnaire dict or None
    """
    try:
        questionnaire = await db_instance.db.questionnaires.find_one({"questionnaire_id": questionnaire_id})
        if questionnaire:
            questionnaire.pop("_id", None)
        return questionnaire
    except Exception as e:
        logger.error(f"❌ Failed to get questionnaire: {e}")
        return None


async def list_questionnaires(limit: int = 50, skip: int = 0, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List questionnaires with optional status filter
    
    Args:
        limit: Maximum number to return
        skip: Number to skip
        status: Filter by status (active/archived)
    
    Returns:
        List of questionnaires
    """
    try:
        query = {} if not status else {"status": status}
        cursor = db_instance.db.questionnaires.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)
        
        questionnaires = []
        async for q in cursor:
            q.pop("_id", None)
            questionnaires.append(q)
        return questionnaires
    except Exception as e:
        logger.error(f"❌ Failed to list questionnaires: {e}")
        return []


async def delete_questionnaire(questionnaire_id: str) -> bool:
    """
    Delete a questionnaire
    
    Args:
        questionnaire_id: Questionnaire ID
    
    Returns:
        True if deleted, False otherwise
    """
    try:
        result = await db_instance.db.questionnaires.delete_one({"questionnaire_id": questionnaire_id})
        if result.deleted_count > 0:
            logger.info(f"✅ Deleted questionnaire: {questionnaire_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Failed to delete questionnaire: {e}")
        return False


# ============================================================================
# CRUD Operations for Questionnaire Invitations
# ============================================================================

async def save_invitation(invitation_data: Dict[str, Any]) -> bool:
    """
    Save questionnaire invitation
    
    Args:
        invitation_data: Invitation data with invitation_id and token
    
    Returns:
        True if successful, False otherwise
    """
    try:
        invitation_data["created_at"] = datetime.utcnow()
        
        await db_instance.db.questionnaire_invitations.update_one(
            {"invitation_id": invitation_data["invitation_id"]},
            {"$set": invitation_data},
            upsert=True
        )
        logger.info(f"✅ Saved invitation: {invitation_data['invitation_id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save invitation: {e}")
        return False


async def get_invitation_by_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Get invitation by token
    
    Args:
        token: Unique token string
    
    Returns:
        Invitation dict or None
    """
    try:
        invitation = await db_instance.db.questionnaire_invitations.find_one({"token": token})
        if invitation:
            invitation.pop("_id", None)
        return invitation
    except Exception as e:
        logger.error(f"❌ Failed to get invitation by token: {e}")
        return None


async def mark_invitation_used(token: str) -> bool:
    """
    Mark invitation as used
    
    Args:
        token: Invitation token
    
    Returns:
        True if successful, False otherwise
    """
    try:
        result = await db_instance.db.questionnaire_invitations.update_one(
            {"token": token},
            {"$set": {"used": True, "used_at": datetime.utcnow()}}
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error(f"❌ Failed to mark invitation used: {e}")
        return False


async def get_invitations_by_questionnaire(questionnaire_id: str) -> List[Dict[str, Any]]:
    """
    Get all invitations for a questionnaire
    
    Args:
        questionnaire_id: Questionnaire ID
    
    Returns:
        List of invitations
    """
    try:
        cursor = db_instance.db.questionnaire_invitations.find(
            {"questionnaire_id": questionnaire_id}
        ).sort("sent_at", DESCENDING)
        
        invitations = []
        async for inv in cursor:
            inv.pop("_id", None)
            invitations.append(inv)
        return invitations
    except Exception as e:
        logger.error(f"❌ Failed to get invitations: {e}")
        return []


# ============================================================================
# CRUD Operations for Questionnaire Responses
# ============================================================================

async def save_response(response_data: Dict[str, Any]) -> bool:
    """
    Save questionnaire response
    
    Args:
        response_data: Response data with response_id
    
    Returns:
        True if successful, False otherwise
    """
    try:
        response_data["submitted_at"] = datetime.utcnow()
        
        await db_instance.db.questionnaire_responses.update_one(
            {"response_id": response_data["response_id"]},
            {"$set": response_data},
            upsert=True
        )
        logger.info(f"✅ Saved response: {response_data['response_id']}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save response: {e}")
        return False


async def get_response_by_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Get response by token (check if already submitted)
    
    Args:
        token: Invitation token
    
    Returns:
        Response dict or None
    """
    try:
        response = await db_instance.db.questionnaire_responses.find_one({"token": token})
        if response:
            response.pop("_id", None)
        return response
    except Exception as e:
        logger.error(f"❌ Failed to get response: {e}")
        return None


async def get_responses_by_questionnaire(questionnaire_id: str) -> List[Dict[str, Any]]:
    """
    Get all responses for a questionnaire
    
    Args:
        questionnaire_id: Questionnaire ID
    
    Returns:
        List of responses
    """
    try:
        cursor = db_instance.db.questionnaire_responses.find(
            {"questionnaire_id": questionnaire_id}
        ).sort("submitted_at", DESCENDING)
        
        responses = []
        async for resp in cursor:
            resp.pop("_id", None)
            responses.append(resp)
        return responses
    except Exception as e:
        logger.error(f"❌ Failed to get responses: {e}")
        return []
