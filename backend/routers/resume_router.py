"""
Resume Parsing Router
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import sys
import os
import hashlib

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from backend.services.feature2_bert_ner import ResumeNERExtractor
from backend.services.feature2_resume_ner_v2 import ResumeNERExtractorV2
from backend.utils.document_parser import extract_text_from_file

router = APIRouter(prefix="/api/resume", tags=["Resume"])

# Lazy load services
ner_extractor = None
ner_extractor_v2 = None

def get_ner_extractor():
    global ner_extractor
    if ner_extractor is None:
        ner_extractor = ResumeNERExtractor()
    return ner_extractor

def get_ner_extractor_v2():
    global ner_extractor_v2
    if ner_extractor_v2 is None:
        ner_extractor_v2 = ResumeNERExtractorV2()
    return ner_extractor_v2

# Pydantic models
class ResumeParseRequest(BaseModel):
    resume_text: str
    candidate_name: Optional[str] = None
    save_to_db: bool = True

@router.post("/parse")
async def parse_resume(request: ResumeParseRequest):
    """
    Full resume parsing with NER extraction - extracts all fields including skills, experience, education
    """
    try:
        extractor = get_ner_extractor()
        resume_data = extractor.parse_resume(request.resume_text)
        
        # Generate resume_id
        resume_id = hashlib.md5(request.resume_text.encode()).hexdigest()[:12]
        
        if resume_data.get("status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_data.get("error", "Resume parsing failed")
            )
        
        # Add candidate name if provided
        if request.candidate_name:
            resume_data["candidate_name"] = request.candidate_name
        
        return {
            "success": True,
            "resume_id": resume_id,
            "name": resume_data.get("name"),
            "email": resume_data.get("email_address", resume_data.get("email")),
            "phone": resume_data.get("phone"),
            "skills": resume_data.get("skills", []),
            "primary_skills": resume_data.get("primary_skills", []),
            "experience": {
                "years": resume_data.get("total_experience_(years)", 0),
                "months": resume_data.get("total_experience_(months)", 0),
                "companies": resume_data.get("companies", [])
            },
            "education": resume_data.get("education", []),
            "certifications": resume_data.get("certifications", []),
            "status": resume_data.get("status"),
            "message": resume_data.get("message"),
            "saved_to_db": False
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume reading failed: {str(e)}"
        )

@router.post("/parse-v2")
async def parse_resume_v2(request: ResumeParseRequest):
    """
    Full resume parsing with Resume-Specific BERT NER V2 (yashpwr/resume-ner-bert-v2)
    """
    try:
        extractor = get_ner_extractor_v2()
        resume_data = extractor.parse_resume(request.resume_text, use_advanced=True)
        
        if resume_data.get("extraction_status") == "FAILED":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=resume_data.get("error", "Resume parsing failed")
            )
        
        # Add candidate name if provided
        if request.candidate_name:
            resume_data["candidate_name"] = request.candidate_name
        
        # Build response matching test expectations
        return {
            "success": True,
            "resume_id": hashlib.md5(request.resume_text.encode()).hexdigest()[:12],
            "name": resume_data.get("name", "Unknown"),
            "email": resume_data.get("email_address", "unknown@email.com"),
            "phone": resume_data.get("contact_number", [""])[0] if resume_data.get("contact_number") else "",
            "skills": resume_data.get("primary_skills", []) + resume_data.get("secondary_skills", []),
            "primary_skills": resume_data.get("primary_skills", []),
            "experience": {
                "years": resume_data.get("total_experience_(months)", 0) // 12,
                "months": resume_data.get("total_experience_(months)", 0),
                "companies": [resume_data.get("current_company_name", "Unknown")]
            },
            "education": resume_data.get("education", []),
            "certifications": [],
            "status": resume_data.get("extraction_status", "SUCCESS"),
            "message": f"Parsed with {resume_data.get('model_type', 'Resume-NER-V2')}",
            "saved_to_db": False,
            "model_type": resume_data.get("model_type", "Resume-NER-V2"),
            "entities": resume_data.get("entities", [])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume V2 parsing failed: {str(e)}"
        )

@router.post("/upload")
async def upload_resume(file: UploadFile = File(...), save_to_db: bool = True):
    """
    Upload resume as file - supports PDF, DOCX, DOC, TXT
    Just stores text, NER extraction happens during matching
    """
    try:
        content = await file.read()
        
        # Extract text based on file type
        resume_text = extract_text_from_file(content, file.filename)
        
        if not resume_text or not resume_text.strip():
            raise HTTPException(400, "No text could be extracted from the file")
        
        extractor = get_ner_extractor()
        resume_data = extractor.simple_read(resume_text)
        resume_data["filename"] = file.filename
        
        return {
            "resume_id": resume_data["resume_id"],
            "filename": file.filename,
            "file_type": file.filename.split('.')[-1].upper(),
            "text_length": len(resume_text),
            "name": resume_data["name"],
            "email": resume_data["email"],
            "phone": resume_data["phone"],
            "text": resume_data["text"],
            "status": resume_data["status"],
            "message": resume_data["message"],
            "saved_to_db": False
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume upload failed: {str(e)}"
        )

@router.post("/batch")
async def batch_parse_resumes(files: List[UploadFile] = File(...), save_to_db: bool = True):
    """
    Upload multiple resumes - supports PDF, DOCX, DOC, TXT
    Full NER extraction happens during matching
    """
    results = []
    
    for file in files:
        try:
            content = await file.read()
            
            # Extract text based on file type
            try:
                resume_text = extract_text_from_file(content, file.filename)
            except HTTPException as e:
                results.append({
                    "filename": file.filename,
                    "status": "FAILED",
                    "error": e.detail
                })
                continue
            
            if not resume_text or not resume_text.strip():
                results.append({
                    "filename": file.filename,
                    "status": "FAILED",
                    "error": "No text could be extracted"
                })
                continue
            
            extractor = get_ner_extractor()
            resume_data = extractor.simple_read(resume_text)
            resume_data["filename"] = file.filename
            
            results.append({
                "resume_id": resume_data["resume_id"],
                "filename": file.filename,
                "file_type": file.filename.split('.')[-1].upper(),
                "text_length": len(resume_text),
                "name": resume_data["name"],
                "email": resume_data["email"],
                "phone": resume_data["phone"],
                "status": "SUCCESS",
                "message": resume_data["message"],
                "saved_to_db": False
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