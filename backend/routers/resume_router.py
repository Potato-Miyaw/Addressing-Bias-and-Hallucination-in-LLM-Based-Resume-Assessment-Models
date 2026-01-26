"""
Resume Parsing Router
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature2_bert_ner import ResumeNERExtractor

router = APIRouter(prefix="/api/resume", tags=["Resume"])

# Lazy load extractor
ner_extractor = None

def get_ner_extractor():
    global ner_extractor
    if ner_extractor is None:
        ner_extractor = ResumeNERExtractor()
    return ner_extractor

# Pydantic models
class ResumeParseRequest(BaseModel):
    resume_text: str
    candidate_name: Optional[str] = None

@router.post("/parse")
async def parse_resume(request: ResumeParseRequest):
    """
    Lightweight resume reading - stores text and extracts basic info
    Full NER extraction happens during matching (performance optimization)
    """
    try:
        extractor = get_ner_extractor()
        resume_data = extractor.simple_read(request.resume_text)
        
        if resume_data.get("status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_data.get("error", "Resume parsing failed")
            )
        
        # CRITICAL: Return the full resume_data INCLUDING the 'text' field
        return {
            "success": True,
            "resume_id": resume_data["resume_id"],
            "name": resume_data["name"],
            "email": resume_data["email"],
            "phone": resume_data["phone"],
            "text": resume_data["text"],  # ✅ ADDED THIS
            "status": resume_data["status"],
            "message": resume_data["message"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume reading failed: {str(e)}"
        )

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    """
    Upload resume as file (TXT only for now)
    Stores text, NER extraction happens during matching
    """
    try:
        content = await file.read()
        
        if file.filename.endswith('.txt'):
            resume_text = content.decode('utf-8')
        else:
            raise HTTPException(400, "Only TXT files supported for now")
        
        extractor = get_ner_extractor()
        resume_data = extractor.simple_read(resume_text)
        
        return {
            "resume_id": resume_data["resume_id"],
            "filename": file.filename,
            "name": resume_data["name"],
            "email": resume_data["email"],
            "phone": resume_data["phone"],
            "text": resume_data["text"],  # ✅ INCLUDE TEXT
            "status": resume_data["status"],
            "message": resume_data["message"]
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume upload failed: {str(e)}"
        )

@router.post("/batch")
async def batch_parse_resumes(files: List[UploadFile] = File(...)):
    """
    Upload multiple resumes - lightweight reading
    Full NER extraction happens during matching
    """
    results = []
    
    for file in files:
        try:
            content = await file.read()
            
            if file.filename.endswith('.txt'):
                resume_text = content.decode('utf-8')
            else:
                results.append({
                    "filename": file.filename,
                    "status": "FAILED",
                    "error": "Unsupported file type"
                })
                continue
            
            extractor = get_ner_extractor()
            resume_data = extractor.simple_read(resume_text)
            
            results.append({
                "resume_id": resume_data["resume_id"],
                "filename": file.filename,
                "name": resume_data["name"],
                "email": resume_data["email"],
                "phone": resume_data["phone"],
                "text": resume_data["text"],  # ✅ INCLUDE TEXT
                "status": "SUCCESS",
                "message": resume_data["message"]
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "FAILED",
                "error": str(e)
            })
    
    return {
        "total": len(files),
        "successful": len([r for r in results if r["status"] == "SUCCESS"]),
        "failed": len([r for r in results if r["status"] == "FAILED"]),
        "results": results
    }