"""
Job Description Extraction Router
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
import sys
import os
import logging

logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature1_jd_extractor import JDExtractor
from backend.utils.document_parser import extract_text_from_file
from backend.utils.id_generator import generate_job_id
from backend.database import (
    save_job_description,
    get_job_description,
    get_job_by_content_hash
)

router = APIRouter(prefix="/api/jd", tags=["Job Description"])

# Lazy load services
jd_extractor = None

def get_jd_extractor():
    global jd_extractor
    if jd_extractor is None:
        jd_extractor = JDExtractor()
    return jd_extractor

# Pydantic models
class JDExtractionRequest(BaseModel):
    jd_text: str
    job_title: Optional[str] = None
    save_to_db: bool = True  # Auto-save by default

class JDExtractionResponse(BaseModel):
    job_id: str
    job_title: Optional[str]
    required_skills: List[str]
    required_experience: int
    required_education: str
    certifications: List[str]
    status: str
    saved_to_db: bool

@router.post("/extract", response_model=JDExtractionResponse)
async def extract_job_description(request: JDExtractionRequest):
    """Extract structured data from job description text and save to DB"""
    try:
        extractor = get_jd_extractor()
        jd_data = extractor.extract_jd_data(request.jd_text)
        
        # Generate hybrid ID (unique + content hash for duplicate detection)
        job_id, content_hash = generate_job_id(request.jd_text, request.job_title)
        
        # Check for duplicates if saving to DB
        is_duplicate = False
        existing_job = None
        if request.save_to_db:
            existing_job = await get_job_by_content_hash(content_hash)
            if existing_job:
                is_duplicate = True
                logger.info(f"⚠️ Duplicate job detected: {existing_job['job_id']}")
        
        # Prepare full job data
        full_job_data = {
            "job_id": job_id,
            "content_hash": content_hash,
            "job_title": request.job_title,
            "jd_text": request.jd_text,
            "required_skills": jd_data["required_skills"],
            "required_experience": jd_data["required_experience"],
            "required_education": jd_data["required_education"],
            "certifications": jd_data["certifications"],
            "extraction_status": jd_data["status"]
        }
        
        # Save to database if requested and not a duplicate
        saved = False
        if request.save_to_db and not is_duplicate:
            saved = await save_job_description(full_job_data)
        
        response_data = {
            "job_id": existing_job["job_id"] if is_duplicate else job_id,
            "job_title": request.job_title,
            "required_skills": jd_data["required_skills"],
            "required_experience": jd_data["required_experience"],
            "required_education": jd_data["required_education"],
            "certifications": jd_data["certifications"],
            "status": jd_data["status"],
            "saved_to_db": saved
        }
        
        # Add duplicate warning if applicable
        if is_duplicate:
            response_data["duplicate_warning"] = f"Job already exists with ID: {existing_job['job_id']}"
        
        return JDExtractionResponse(**response_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"JD extraction failed: {str(e)}"
        )

@router.post("/upload")
async def upload_job_description(file: UploadFile = File(...), save_to_db: bool = True):
    """
    Upload JD as file - supports PDF, DOCX, DOC, TXT
    
    Extracts text from the file and processes it for structured data extraction
    """
    try:
        content = await file.read()
        
        # Extract text based on file type
        jd_text = extract_text_from_file(content, file.filename)
        
        if not jd_text or not jd_text.strip():
            raise HTTPException(400, "No text could be extracted from the file")
        
        extractor = get_jd_extractor()
        jd_data = extractor.extract_jd_data(jd_text)
        
        # Generate hybrid ID
        job_title = file.filename.rsplit('.', 1)[0]  # Remove extension
        job_id, content_hash = generate_job_id(jd_text, job_title)
        
        # Check for duplicates
        is_duplicate = False
        existing_job = None
        if save_to_db:
            existing_job = await get_job_by_content_hash(content_hash)
            if existing_job:
                is_duplicate = True
                logger.info(f"⚠️ Duplicate job detected: {existing_job['job_id']}")
        
        # Prepare full job data
        full_job_data = {
            "job_id": job_id,
            "content_hash": content_hash,
            "job_title": job_title,
            "jd_text": jd_text,
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1].upper(),
            "required_skills": jd_data["required_skills"],
            "required_experience": jd_data["required_experience"],
            "required_education": jd_data["required_education"],
            "certifications": jd_data["certifications"],
            "extraction_status": jd_data["status"]
        }
        
        # Save to database if requested and not duplicate
        saved = False
        if save_to_db and not is_duplicate:
            saved = await save_job_description(full_job_data)
        
        result = {
            "job_id": existing_job["job_id"] if is_duplicate else job_id,
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1].upper(),
            "text_length": len(jd_text),
            "jd_data": jd_data,
            "saved_to_db": saved
        }
        
        if is_duplicate:
            result["duplicate_warning"] = f"Job already exists with ID: {existing_job['job_id']}"
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )