"""
Job Description Extraction Router
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature1_jd_extractor import JDExtractor

router = APIRouter(prefix="/api/jd", tags=["Job Description"])

# Lazy load extractor
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

class JDExtractionResponse(BaseModel):
    job_id: str
    job_title: Optional[str]
    required_skills: List[str]
    required_experience: int
    required_education: str
    certifications: List[str]
    status: str

@router.post("/extract", response_model=JDExtractionResponse)
async def extract_job_description(request: JDExtractionRequest):
    """Extract structured data from job description text"""
    try:
        extractor = get_jd_extractor()
        jd_data = extractor.extract_jd_data(request.jd_text)
        
        import hashlib
        job_id = hashlib.md5(request.jd_text.encode()).hexdigest()[:12]
        
        return JDExtractionResponse(
            job_id=job_id,
            job_title=request.job_title,
            required_skills=jd_data["required_skills"],
            required_experience=jd_data["required_experience"],
            required_education=jd_data["required_education"],
            certifications=jd_data["certifications"],
            status=jd_data["status"]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"JD extraction failed: {str(e)}"
        )

@router.post("/upload")
async def upload_job_description(file: UploadFile = File(...)):
    """Upload JD as file (TXT only for now)"""
    try:
        content = await file.read()
        
        if file.filename.endswith('.txt'):
            jd_text = content.decode('utf-8')
        else:
            raise HTTPException(400, "Only TXT files supported for now")
        
        extractor = get_jd_extractor()
        jd_data = extractor.extract_jd_data(jd_text)
        
        import hashlib
        job_id = hashlib.md5(jd_text.encode()).hexdigest()[:12]
        
        return {
            "job_id": job_id,
            "filename": file.filename,
            "jd_data": jd_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )